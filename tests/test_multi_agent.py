from __future__ import annotations
import json, os, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from agent_system import PlannerAgent, EvaluatorAgent, SynthesizerAgent, AgentMessage, MultiAgentOrchestrator
class T(unittest.TestCase):
    def test_01_plan(self):
        p=PlannerAgent().run("MCP"); self.assertEqual(p.role,"planner"); self.assertFalse(p.payload["raw_query_persisted"])
    def test_02_roles(self):
        self.assertEqual(MultiAgentOrchestrator().run("stdio JSON RPC MCP tools")["roles"],["planner","researcher","evaluator","synthesizer"])
    def test_03_research(self):
        self.assertEqual(MultiAgentOrchestrator().run("stdio JSON RPC MCP tools resources prompts")["research"]["evidence"][0]["doc_id"],"mcp_server")
    def test_04_eval(self):
        self.assertTrue(MultiAgentOrchestrator().run("query hash telemetry raw query privacy")["evaluation"]["sufficient_evidence"])
    def test_05_cites(self):
        self.assertGreaterEqual(len(MultiAgentOrchestrator().run("recall MRR evaluation")["synthesis"]["cited_doc_ids"]),1)
    def test_06_abstain(self):
        r=AgentMessage("researcher",{"evidence":[{"doc_id":"x","score":0.0,"text":"x"}]}); e=EvaluatorAgent().run(r); s=SynthesizerAgent().run(r,e)
        self.assertTrue(s.payload["abstained"])
    def test_07_policy(self):
        self.assertEqual(MultiAgentOrchestrator().run("evidence discipline")["evaluation"]["claim_policy"],"ClaimStrength <= EvidenceStrength")
    def test_08_invalid(self):
        with self.assertRaises(ValueError): MultiAgentOrchestrator().run("")
    def test_09_cli(self):
        p=subprocess.run([sys.executable,str(ROOT/"agent_system.py"),"--query","voice study coach"],capture_output=True,text=True,timeout=30,env=os.environ.copy())
        self.assertEqual(p.returncode,0); self.assertEqual(json.loads(p.stdout)["trace_length"],4)
    def test_10_no_network(self):
        t=(ROOT/"agent_system.py").read_text().lower(); self.assertNotIn("import requests",t); self.assertNotIn("import socket",t)
if __name__=="__main__": unittest.main(verbosity=2)
