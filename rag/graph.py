"""LangGraphによるRAGフロー制御（検索→判定→再検索）"""

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from rag.llm import build_llm
from rag.retriever import retrieve

MAX_RETRIES = 2


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    is_sufficient: bool
    retry_count: int


def _build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """以下のコンテキストを使って質問に答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
    )


def _build_judge_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """以下の回答は質問に十分答えていますか？「はい」または「いいえ」のみ答えてください。

質問: {question}
回答: {answer}"""
    )


def build_graph(vectorstore: Chroma) -> StateGraph:
    llm = build_llm()
    answer_chain = _build_answer_prompt() | llm | StrOutputParser()
    judge_chain = _build_judge_prompt() | llm | StrOutputParser()

    def search_node(state: RAGState) -> RAGState:
        docs = retrieve(vectorstore, state["question"])
        state["context"] = "\n\n".join(doc.page_content for doc in docs)
        return state

    def generate_node(state: RAGState) -> RAGState:
        state["answer"] = answer_chain.invoke(
            {"context": state["context"], "question": state["question"]}
        )
        return state

    def judge_node(state: RAGState) -> RAGState:
        result = judge_chain.invoke(
            {"question": state["question"], "answer": state["answer"]}
        )
        state["is_sufficient"] = "はい" in result
        # retry_countはノード内でインクリメントすることで確実にグラフ状態へ反映される
        state["retry_count"] = state["retry_count"] + 1
        return state

    def should_retry(state: RAGState) -> str:
        # ルーティング関数はstate変更不可のためカウント判定のみ行う
        if state["is_sufficient"] or state["retry_count"] >= MAX_RETRIES:
            return "end"
        return "retry"

    graph = StateGraph(RAGState)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "judge")
    graph.add_conditional_edges(
        "judge",
        should_retry,
        {"end": END, "retry": "search"},
    )

    return graph.compile()


def run_graph(vectorstore: Chroma, question: str) -> str:
    """グラフを実行して回答を返す"""
    app = build_graph(vectorstore)
    final_state = app.invoke(
        {"question": question, "context": "", "answer": "", "is_sufficient": False, "retry_count": 0}
    )
    return final_state["answer"]
