"""Step 3 - RAGAS evaluation of V1 vs V2 prompts."""

from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

import config
config.assert_ready()
config.enable_langsmith_tracing()

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from qa_pairs import QA_PAIRS

from importlib import import_module
_step1 = import_module("01_langsmith_rag_pipeline")
build_vectorstore = _step1.build_vectorstore
make_llm = _step1.make_llm
make_embeddings = _step1.make_embeddings

_step2 = import_module("02_prompt_hub_ab_routing")
PROMPT_V1 = _step2.PROMPT_V1
PROMPT_V2 = _step2.PROMPT_V2

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": ctx_str, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list[dict]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()
    prompt = PROMPTS[prompt_version]

    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    results: list[dict] = []
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")
    return results


def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list[dict], version: str) -> dict:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)

    llm_eval = LangchainLLMWrapper(make_llm())
    emb_eval = LangchainEmbeddingsWrapper(make_embeddings())

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        col = df[key].dropna().tolist()
        scores[key] = float(np.mean(col)) if col else float("nan")

    for k, v in scores.items():
        star = " *" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


def main() -> None:
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 60)
    print("  V1 vs V2 comparison")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "<- V1" if s1 > s2 else ("<- V2" if s2 > s1 else "tie")
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nTarget met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\nBelow target ({best_faith:.4f}). Adjust chunking or prompts and retry.")

    config.DATA_DIR.mkdir(exist_ok=True)
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "best_faithfulness": best_faith,
    }
    config.RAGAS_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Saved {config.RAGAS_REPORT}")


if __name__ == "__main__":
    main()
