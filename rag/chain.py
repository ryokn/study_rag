"""LangChainを使ったRAGチェーンの定義"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.llm import build_llm

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """以下のコンテキストのみを使って質問に答えてください。
コンテキストに答えが含まれない場合は「わかりません」と答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
)


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: Chroma) -> Runnable:
    """RAGチェーンを構築して返す"""
    llm = build_llm()
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain
