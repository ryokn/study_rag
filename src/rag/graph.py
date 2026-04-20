"""LangGraphによるRAGフロー制御（検索→生成→判定→再検索）

LangGraphの基本概念:
  - StateGraph: 状態（State）を持つグラフ。各ノードが状態を受け取り、
                更新した新しい状態を返す。
  - ノード: グラフの処理単位。state を受け取り、更新した state を返す関数。
  - エッジ: ノード間の遷移。条件分岐（conditional_edges）も使用可能。

イミュータビリティ原則:
  state を直接変更する（state["key"] = value）のではなく、
  スプレッド演算子相当の {**state, "key": value} で新しい dict を返す。
  これにより副作用を防ぎ、デバッグが容易になる。
"""

from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from rag.llm import build_llm
from rag.retriever import retrieve

# 回答品質が不十分な場合の最大再試行回数
MAX_RETRIES = 2


class RAGState(TypedDict):
    """グラフ全体で共有される状態の型定義。

    TypedDict を使うことで、各フィールドの型が明確になりIDEの補完も効く。
    """

    question: str                    # ユーザーの質問文
    context: str                     # ChromaDBから取得したチャンクを結合した文字列
    answer: str                      # LLMが生成した回答
    is_sufficient: bool              # 回答が十分かどうかのLLM判定結果
    retry_count: int                 # 再試行回数（MAX_RETRIESを超えたら強制終了）
    history: list[tuple[str, str]]   # 会話履歴 [(質問, 回答), ...]


def _format_history(history: list[tuple[str, str]]) -> str:
    """会話履歴をプロンプトに埋め込む形式の文字列に変換する。

    履歴がない場合は空文字列を返すことで、初回質問に余計なコンテキストを
    付与しないようにしている。
    """
    if not history:
        return ""
    lines = ["【過去の会話】"]
    for q, a in history:
        lines.append(f"ユーザー: {q}")
        lines.append(f"アシスタント: {a}")
    # 末尾に空行を入れて現在の質問と区切る
    return "\n".join(lines) + "\n\n"


def _build_answer_prompt() -> ChatPromptTemplate:
    """回答生成用プロンプトテンプレートを返す。

    {history}には過去の会話、{context}にはPDF検索結果、{question}には
    現在の質問が埋め込まれる。
    """
    return ChatPromptTemplate.from_template(
        """{history}以下のコンテキストを使って質問に答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
    )


def _build_judge_prompt() -> ChatPromptTemplate:
    """回答品質判定用プロンプトテンプレートを返す。

    LLMに「はい/いいえ」のみ答えさせることで、後続処理での
    文字列判定を単純にできる。
    """
    return ChatPromptTemplate.from_template(
        """以下の回答は質問に十分答えていますか？「はい」または「いいえ」のみ答えてください。

質問: {question}
回答: {answer}"""
    )


def build_graph(vectorstore: Chroma) -> StateGraph:
    """RAGフローのLangGraphを構築してコンパイル済みグラフを返す。

    グラフ構造:
      search → generate → judge → (十分 or 上限) → END
                              └─→ (不十分) → search (再試行)
    """
    llm = build_llm()
    answer_chain = _build_answer_prompt() | llm | StrOutputParser()
    judge_chain = _build_judge_prompt() | llm | StrOutputParser()

    def search_node(state: RAGState) -> RAGState:
        """ChromaDBから関連チャンクを検索してcontextに格納するノード。

        イミュータビリティ: state を直接変更せず、新しい dict を返す。
        """
        docs = retrieve(vectorstore, state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {**state, "context": context}

    def generate_node(state: RAGState) -> RAGState:
        """コンテキストと質問をLLMに渡して回答を生成するノード。"""
        answer = answer_chain.invoke(
            {
                "history": _format_history(state["history"]),
                "context": state["context"],
                "question": state["question"],
            }
        )
        return {**state, "answer": answer}

    def judge_node(state: RAGState) -> RAGState:
        """生成した回答が十分かどうかをLLMに判定させるノード。

        retry_count をここでインクリメントすることで、
        should_retry のルーティングで正しい回数を参照できる。
        """
        result = judge_chain.invoke(
            {"question": state["question"], "answer": state["answer"]}
        )
        return {
            **state,
            "is_sufficient": "はい" in result,
            "retry_count": state["retry_count"] + 1,
        }

    def should_retry(state: RAGState) -> str:
        """判定結果に基づいて次のノードを決定するルーティング関数。

        ルーティング関数は状態を変更せず、遷移先のキー文字列のみを返す。
        """
        if state["is_sufficient"] or state["retry_count"] >= MAX_RETRIES:
            return "end"
        return "retry"

    # グラフにノードを登録
    graph = StateGraph(RAGState)
    graph.add_node("search", search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("judge", judge_node)

    # エントリーポイントと通常エッジを設定
    graph.set_entry_point("search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", "judge")

    # 条件付きエッジ: should_retry の戻り値によって遷移先を振り分ける
    graph.add_conditional_edges(
        "judge",
        should_retry,
        {"end": END, "retry": "search"},
    )

    # compile() でグラフを実行可能な状態に変換する
    return graph.compile()


def run_graph(
    vectorstore: Chroma,
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """グラフを実行して最終的な回答文字列を返す。

    初期状態として全フィールドを明示的に指定する。
    LangGraphはこの状態をノード間で引き渡しながら更新していく。
    """
    app = build_graph(vectorstore)
    final_state = app.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "is_sufficient": False,
            "retry_count": 0,
            "history": history or [],
        }
    )
    return final_state["answer"]
