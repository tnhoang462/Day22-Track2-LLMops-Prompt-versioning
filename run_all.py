"""Run all four lab steps sequentially, or a single step with --step N."""

from __future__ import annotations

import argparse
import importlib
import sys
import time

STEPS = {
    1: ("01_langsmith_rag_pipeline", "LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing",  "Prompt Hub & A/B Routing"),
    3: ("03_ragas_evaluation",       "RAGAS Evaluation (slow)"),
    4: ("04_guardrails_validator",   "Guardrails Validators"),
}


def run_step(step_num: int) -> None:
    module_name, label = STEPS[step_num]
    print("\n" + "#" * 60)
    print(f"#  STEP {step_num}: {label}")
    print("#" * 60)
    start = time.time()
    module = importlib.import_module(module_name)
    module.main()
    elapsed = time.time() - start
    print(f"\n>> Step {step_num} done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 22 lab runner")
    parser.add_argument("--step", type=int, choices=sorted(STEPS), help="Run only this step")
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else sorted(STEPS)
    for step in steps_to_run:
        try:
            run_step(step)
        except Exception as e:
            print(f"!! Step {step} failed: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
