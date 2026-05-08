"""Step 2 - Push two prompt versions to LangSmith Prompt Hub and route 50/50."""

from __future__ import annotations

import hashlib

import config
config.assert_ready()
config.enable_langsmith_tracing()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS

from importlib import import_module
_step1 = import_module("01_langsmith_rag_pipeline")
build_vectorstore = _step1.build_vectorstore
make_llm = _step1.make_llm


SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say "
    "'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"


def push_prompts_to_hub(client: Client) -> None:
    for name, tmpl, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 - concise 2-4 sentence answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 - structured 3-5 sentence answers"),
    ]:
        try:
            url = client.push_prompt(name, object=tmpl, description=desc)
            print(f"Pushed {name} -> {url}")
        except Exception as e:
            print(f"Push failed for {name}: {e}")


def pull_prompts_from_hub(client: Client) -> dict:
    prompts: dict = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"Pulled '{name}' from Hub")
        except Exception as e:
            prompts[name] = fallback
            print(f"Using local fallback for '{name}' ({e})")
    return prompts


def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if h % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


def main() -> None:
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    print("\n-- Push prompts to Hub --")
    push_prompts_to_hub(client)

    print("\n-- Pull prompts from Hub --")
    prompts = pull_prompts_from_hub(client)

    print("\n-- Build retriever + LLM --")
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()

    print("\n-- Run 50 questions through A/B router --")
    counts = {"v1": 0, "v2": 0}
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        counts[version_tag] += 1
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}")
        print(f"      -> {result['answer'][:100]}")

    print("\n-- Routing summary --")
    print(f"  V1: {counts['v1']} requests")
    print(f"  V2: {counts['v2']} requests")
    print(f"  Total: {sum(counts.values())} traces sent to LangSmith")


if __name__ == "__main__":
    main()
