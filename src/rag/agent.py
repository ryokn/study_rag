"""LangGraphのReActエージェント（PDF検索・Web検索・計算・コード実行）"""

import math
import os

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from rag.llm import build_base_llm
from rag.retriever import retrieve

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


def _build_tools(vectorstore: Chroma) -> list:
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
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)

    @tool
    def calculator(expression: str) -> str:
        """数式を計算する。四則演算や数学関数が使える（例: '2 + 3 * 4', 'sqrt(16)', 'pi * 2'）。"""
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        try:
            result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"計算エラー: {e}"

    @tool
    def python_repl(code: str) -> str:
        """Pythonコードを実行する。データ処理や複雑な計算に使う。ローカル実行のみ。"""
        from langchain_community.tools.python.tool import PythonREPLTool
        return PythonREPLTool().run(code)

    return [search_pdf, web_search, calculator, python_repl]


def build_agent(vectorstore: Chroma):
    """ReActエージェントを構築して返す"""
    llm = build_base_llm()
    tools = _build_tools(vectorstore)
    return create_react_agent(llm, tools)


def run_agent(
    vectorstore: Chroma,
    question: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """エージェントを実行して回答を返す"""
    agent = build_agent(vectorstore)

    messages = []
    for q, a in (history or []):
        messages.extend([HumanMessage(content=q), AIMessage(content=a)])
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})
    content = result["messages"][-1].content
    # Anthropicモデルはcontentがリスト形式のブロックで返る場合がある
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content
