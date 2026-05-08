"""Step 1 - LangSmith-instrumented RAG pipeline.

Builds a FAISS-backed RAG chain over data/knowledge_base.txt and runs all
50 sample questions through @traceable so each call shows up in LangSmith.
"""

from __future__ import annotations

import config
config.assert_ready()
config.enable_langsmith_tracing()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=0.0,
    )


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


def build_vectorstore() -> FAISS:
    text = config.KNOWLEDGE_BASE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split knowledge base into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, make_embeddings())


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the user's question using ONLY the "
     "provided context. If the context does not contain the answer, say "
     "'I don't have enough information.'\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main() -> None:
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:120]}\n")

    print(f"Sent {len(SAMPLE_QUESTIONS)} traces to LangSmith project "
          f"'{config.LANGSMITH_PROJECT}'.")
    print("Open https://smith.langchain.com to view them.")


if __name__ == "__main__":
    main()
