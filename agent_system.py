#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

def rag_root():
    raw=os.environ.get("NALLAR_RAG_PACKAGE")
    return Path(raw).expanduser().resolve() if raw else (Path(__file__).resolve().parents[1]/"03_RAG_TELEMETRY_EVALS").resolve()

def load_rag():
    path=rag_root()/"rag_engine.py"
    if not path.is_file(): raise FileNotFoundError(str(path))
    spec=importlib.util.spec_from_file_location("nallar_g3_rag",path)
    if spec is None or spec.loader is None: raise RuntimeError("unable to load RAG module")
    mod=importlib.util.module_from_spec(spec); __import__("sys").modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

@dataclass
class AgentMessage:
    role:str
    payload:Dict[str,Any]

class PlannerAgent:
    def run(self,q):
        return AgentMessage("planner",{"query_sha256":hashlib.sha256(q.encode()).hexdigest(),
            "tasks":["retrieve evidence","evaluate sufficiency","synthesize bounded answer"],"raw_query_persisted":False})

class ResearcherAgent:
    def __init__(self,corpus=None,k=3):
        rag=load_rag(); self.svc=rag.RAGService(Path(corpus) if corpus else rag_root()/"data/sample_corpus.json"); self.k=k
    def run(self,q):
        rows=self.svc.retrieve(q,self.k)
        return AgentMessage("researcher",{"evidence":[{"rank":r["rank"],"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],
            "title":r["title"],"score":r["score"],"text":r["text"]} for r in rows]})

class EvaluatorAgent:
    def run(self,research):
        ev=research.payload.get("evidence") or []
        positive=[x for x in ev if float(x.get("score",0))>0]
        return AgentMessage("evaluator",{"sufficient_evidence":bool(positive),"positive_evidence_count":len(positive),
            "claim_policy":"ClaimStrength <= EvidenceStrength","abstain_if_no_positive_evidence":True})

class SynthesizerAgent:
    def run(self,research,evaluation):
        ev=research.payload.get("evidence") or []
        if not evaluation.payload["sufficient_evidence"]:
            return AgentMessage("synthesizer",{"answer":"Insufficient retrieved evidence; no substantive answer asserted.",
                "cited_doc_ids":[],"abstained":True,"claim_policy":"ClaimStrength <= EvidenceStrength"})
        top=ev[:2]
        return AgentMessage("synthesizer",{"answer":"Evidence summary: "+" | ".join("["+x["doc_id"]+"] "+x["text"] for x in top),
            "cited_doc_ids":[x["doc_id"] for x in top],"abstained":False,"claim_policy":"ClaimStrength <= EvidenceStrength"})

class MultiAgentOrchestrator:
    def __init__(self,corpus=None,k=3):
        self.planner=PlannerAgent(); self.researcher=ResearcherAgent(corpus,k); self.evaluator=EvaluatorAgent(); self.synthesizer=SynthesizerAgent()
    def run(self,q):
        if not isinstance(q,str) or not q.strip(): raise ValueError("query required")
        p=self.planner.run(q); r=self.researcher.run(q); e=self.evaluator.run(r); s=self.synthesizer.run(r,e)
        return {"roles":["planner","researcher","evaluator","synthesizer"],"plan":p.payload,"research":r.payload,
                "evaluation":e.payload,"synthesis":s.payload,"trace_length":4}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--query",required=True); ap.add_argument("--corpus",default=None); ap.add_argument("--k",type=int,default=3)
    a=ap.parse_args(); print(json.dumps(MultiAgentOrchestrator(a.corpus,a.k).run(a.query),indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
