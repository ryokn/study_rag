"""マルチエージェント実装（Supervisor パターン）

このモジュールは LangGraph の Supervisor パターンを使ったマルチエージェントを実装する。

【シングルエージェント（agent.py）との違い】
  agent.py    : 1つのLLMが全ツールを自律的に使い回す（ReAct）
  このファイル : 複数のLLMが役割分担して協調する（Supervisor）

【エージェント構成】
  Supervisor（司令塔）
    ├─ ResearchAgent: PDF検索・Web検索に特化。情報収集のみを担当
    └─ AnswerAgent  : 収集済み情報をもとに最終回答を生成

【処理フロー】
  ユーザーの質問
      ↓
  Supervisor（質問を分析し「調査が必要か？」を判断）
      ├─ 調査が必要 → ResearchAgent（PDF + Web検索）→ Supervisor へ戻る
      └─ 調査不要  → AnswerAgent（直接回答生成）
                                ↓
                          AnswerAgent（研究結果をもとに回答生成）
                                ↓
                           最終回答を返す

【LangGraph の StateGraph について】
  各ノードは MultiAgentState を受け取り、更新した state を返す。
  Supervisor ノードの `next` フィールドが次のノードへのルーティングを決定する。
"""

import os
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from rag.llm import build_base_llm
from rag.retriever import retrieve

# Supervisor が次のエージェントとして選べる選択肢
RESEARCH = "research"
ANSWER = "answer"
FINISH = "FINISH"


class MultiAgentState(TypedDict):
    """マルチエージェント全体で共有される状態。

    各ノードはこの state を受け取り、更新した state を返す。
    イミュータビリティ原則に従い {**state, "key": value} で更新する。
    """

    question: str                      # ユーザーの質問
    research_result: str               # ResearchAgent が収集した情報
    answer: str                        # AnswerAgent が生成した最終回答
    history: list[tuple[str, str]]     # 会話履歴 [(質問, 回答), ...]
    next: str                          # 次に呼ぶノード: research / answer / FINISH


def _build_research_tools(vectorstore: Chroma) -> list:
    """ResearchAgent が使用するツール（PDF検索・Web検索）を返す。

    ResearchAgent は「情報収集のみ」に集中するため、
    計算やコード実行などのツールは持たない。
    """

    @tool
    def search_pdf(query: str) -> str:
        """PDFドキュメントから情報を検索する。ドキュメントの内容に関する質問に使う。"""
        docs = retrieve(vectorstore, query)
        if not docs:
            return "関連するドキュメントが見つかりませんでした。"
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def web_search(query: str) -> str:
        """インターネットで最新情報を検索する。リアルタイム情報や一般知識に使う。"""
        return DuckDuckGoSearchRun().run(query)

    return [search_pdf, web_search]


def _build_supervisor_chain():
    """Supervisor の判断ロジック（プロンプトチェーン）を返す。

    Supervisor は質問と調査結果を受け取り、
    次に呼ぶエージェントを "research" / "answer" / "FINISH" のいずれかで返す。

    LLMに選択肢を1単語で返させることで、後続のルーティング処理を単純にできる。
    """
    prompt = ChatPromptTemplate.from_template(
        """あなたはマルチエージェントの司令塔（Supervisor）です。
以下の情報をもとに、次に実行すべきエージェントを1単語で返してください。

選択肢:
- research : PDF・Web検索が必要な場合
- answer   : 十分な情報が揃っており回答を生成できる場合
- FINISH   : 回答が完成している場合

質問: {question}
調査結果: {research_result}
現在の回答: {answer}

次のエージェント（research / answer / FINISH のいずれか1単語のみ）:"""
    )
    return prompt | build_base_llm() | StrOutputParser()


def build_multi_agent_graph(vectorstore: Chroma) -> StateGraph:
    """Supervisor パターンのマルチエージェントグラフを構築して返す。

    グラフ構造:
      supervisor → research（調査が必要な場合）
      supervisor → answer（回答生成に移る場合）
      supervisor → END（完了の場合）
      research   → supervisor（調査結果を報告して次の指示を仰ぐ）
      answer     → END（回答生成後は終了）
    """
    llm = build_base_llm()
    research_tools = _build_research_tools(vectorstore)
    # create_react_agent: LangGraph 組み込みの ReAct エージェントファクトリ
    # ResearchAgent は ReAct パターンで自律的にツールを選択・実行する
    research_agent = create_react_agent(llm, research_tools)
    supervisor_chain = _build_supervisor_chain()

    def supervisor_node(state: MultiAgentState) -> MultiAgentState:
        """質問と現在の調査結果を見て次のエージェントを決定する。

        research_result が空の場合は必ず research を選択することで、
        初回は必ず情報収集を行うように制御する。
        """
        # 調査がまだの場合は強制的に research へ
        if not state["research_result"]:
            return {**state, "next": RESEARCH}

        raw = supervisor_chain.invoke(
            {
                "question": state["question"],
                "research_result": state["research_result"],
                "answer": state["answer"],
            }
        ).strip().lower()

        # LLM の出力から next を決定（予期しない出力は answer にフォールバック）
        if RESEARCH in raw:
            next_node = RESEARCH
        elif FINISH in raw.upper():
            next_node = FINISH
        else:
            next_node = ANSWER

        return {**state, "next": next_node}

    def research_node(state: MultiAgentState) -> MultiAgentState:
        """ResearchAgent を実行して調査結果を state に格納する。

        会話履歴を HumanMessage / AIMessage に変換して渡すことで
        マルチターン対話にも対応する。
        """
        messages: list = []
        for q, a in state["history"]:
            messages.extend([HumanMessage(content=q), AIMessage(content=a)])
        messages.append(HumanMessage(content=state["question"]))

        result = research_agent.invoke({"messages": messages})
        # ReAct エージェントの最終メッセージが調査結果
        research_result = result["messages"][-1].content

        if isinstance(research_result, list):
            research_result = "".join(
                block.get("text", "") for block in research_result if isinstance(block, dict)
            )

        return {**state, "research_result": research_result}

    def answer_node(state: MultiAgentState) -> MultiAgentState:
        """AnswerAgent: 調査結果をもとに最終回答を生成する。

        AnswerAgent はツールを持たず、収集済みの research_result だけを使って
        簡潔で正確な回答を生成することに特化している。
        """
        prompt = ChatPromptTemplate.from_template(
            """あなたは収集された情報をもとに、簡潔で正確な回答を生成する専門家です。

調査結果:
{research_result}

質問: {question}

上記の調査結果のみを根拠にして、質問に答えてください。
調査結果に含まれていない情報は使わないでください。

回答:"""
        )
        chain = prompt | build_base_llm() | StrOutputParser()
        answer = chain.invoke(
            {
                "research_result": state["research_result"],
                "question": state["question"],
            }
        )
        return {**state, "answer": answer, "next": FINISH}

    def route(state: MultiAgentState) -> str:
        """Supervisor の next フィールドに基づいて遷移先を返すルーティング関数。

        LangGraph の conditional_edges はこの関数の戻り値で分岐先を決定する。
        """
        if state["next"] == RESEARCH:
            return RESEARCH
        if state["next"] == ANSWER:
            return ANSWER
        return END

    # グラフ構築
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("supervisor")

    # Supervisor からの条件付きエッジ
    graph.add_conditional_edges(
        "supervisor",
        route,
        {RESEARCH: "research", ANSWER: "answer", END: END},
    )
    # ResearchAgent は完了後 Supervisor に戻り次の指示を仰ぐ
    graph.add_edge("research", "supervisor")
    # AnswerAgent は完了後そのまま終了
    graph.add_edge("answer", END)

    return graph.compile()


def run_multi_agent(
    vectorstore: Chroma,
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """マルチエージェントを実行して最終回答を返す。

    Args:
        vectorstore: PDF検索に使用するChromaDBインスタンス
        question: ユーザーの質問文
        history: 過去の会話履歴 [(質問, 回答), ...]

    Returns:
        AnswerAgent が生成した最終回答文字列
    """
    app = build_multi_agent_graph(vectorstore)
    final_state = app.invoke(
        {
            "question": question,
            "research_result": "",
            "answer": "",
            "history": history or [],
            "next": "",
        }
    )
    return final_state["answer"]
