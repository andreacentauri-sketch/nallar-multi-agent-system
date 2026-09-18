# Multi-Agent System Design

Planner -> Researcher -> Evaluator -> Synthesizer. Researcher uses the verified G2 RAG package. Evaluator enforces `ClaimStrength <= EvidenceStrength` and the synthesizer abstains when no positive evidence exists. This demonstrates deterministic orchestration, not open-world autonomy.
