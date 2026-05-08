"""Centralized configuration loader for the Day 22 lab."""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")
LANGSMITH_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DATA_DIR = ROOT / "data"
KNOWLEDGE_BASE = DATA_DIR / "knowledge_base.txt"
RAGAS_REPORT = DATA_DIR / "ragas_report.json"


def enable_langsmith_tracing() -> None:
    """Set the env vars LangChain needs to emit traces. Call BEFORE importing langchain."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT


def assert_ready() -> None:
    missing = []
    if not LANGSMITH_API_KEY:
        missing.append("LANGCHAIN_API_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing env vars: {', '.join(missing)}. Fill them in .env then re-run."
        )


if __name__ == "__main__":
    print("Config loaded successfully")
    print(f"   LangSmith project : {LANGSMITH_PROJECT}")
    print(f"   OpenAI endpoint   : {OPENAI_BASE_URL}")
    print(f"   Default LLM model : {LLM_MODEL}")
    print(f"   Embedding model   : {EMBEDDING_MODEL}")
    print(f"   LangSmith key set : {bool(LANGSMITH_API_KEY)}")
    print(f"   OpenAI key set    : {bool(OPENAI_API_KEY)}")
