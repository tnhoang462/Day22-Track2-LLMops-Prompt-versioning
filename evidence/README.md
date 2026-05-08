# Evidence — Day 22 Lab Submission

## Results summary

| Task | Status | Key result |
|---|---|---|
| 1 — LangSmith RAG Pipeline | done | 50 traces sent to project `day22-langsmith-lab` |
| 2 — Prompt Hub & A/B Routing | done | 2 prompts pushed (`day22-rag-prompt-v1/v2`) + 50 A/B traces (V1=19, V2=31) |
| 3 — RAGAS Evaluation | done | **V1 faithfulness = 0.9838** (target ≥ 0.8), V2 = 0.8524 |
| 4 — Guardrails Validators | done | 6/6 PII cases redacted, 5/5 JSON cases handled |

## V1 vs V2 analysis (Task 3)

V1 (concise, 2-4 sentences) outperforms V2 (structured, 3-5 sentences) on every metric:

| Metric | V1 | V2 | Winner |
|---|---|---|---|
| faithfulness | **0.9838** | 0.8524 | V1 |
| answer_relevancy | **0.9232** | 0.9022 | V1 |
| context_recall | 1.0000 | 1.0000 | tie |
| context_precision | **0.9700** | 0.9683 | V1 |

**Why V1 wins:** the concise prompt encourages the model to stay strictly within
the retrieved context, while V2's "structured 3-5 sentence" instruction nudges
the model to expand and elaborate, which introduces claims that are not strictly
supported by the chunks — lowering faithfulness by ~13 points. Both prompts share
the same retriever, so context_recall is identical.

## Files

| File | What it contains |
|---|---|
| `01_langsmith_traces.png` | LangSmith UI screenshot (≥ 50 traces, take during/after Task 1) |
| `02_prompt_hub.png` | Prompt Hub UI showing both v1 and v2 prompts |
| `02_ab_routing_log.txt` | Console output of the 50 A/B routed queries |
| `03_ragas_scores.png` | Screenshot of the comparison table |
| `03_ragas_scores.txt` | Plain-text version of the same table |
| `03_ragas_report.json` | Per-version metric scores (copy of `data/ragas_report.json`) |
| `04_pii_demo_log.txt` | PII detector demo on 6 inputs |
| `04_json_demo_log.txt` | JSON formatter demo on 5 inputs |

## To capture the screenshots
1. `01_langsmith_traces.png` → https://smith.langchain.com → project `day22-langsmith-lab` → Runs tab (≥ 100 traces total after step 2)
2. `02_prompt_hub.png` → LangSmith Prompts tab → both `day22-rag-prompt-v1` and `day22-rag-prompt-v2`
3. `03_ragas_scores.png` → run `python 03_ragas_evaluation.py` and screenshot the final comparison block, OR screenshot `03_ragas_scores.txt`
