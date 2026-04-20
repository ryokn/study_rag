"""LangGraphのReActエージェント（PDF検索・Web検索・計算・コード実行）

ReAct（Reasoning + Acting）パターンとは:
  LLMが「考える（Reason）→ ツールを使う（Act）→ 結果を観察（Observe）」を
  繰り返すことで、複雑な質問を段階的に解決するエージェント設計パターン。

  通常のRAGチェーンとの違い:
    RAGチェーン: 検索 → 回答 の固定パイプライン
    ReActエージェント: LLMが自律的にどのツールをいつ使うかを決定する

ツール（@tool デコレータ）:
  LangChainの @tool デコレータを付与した関数はLLMが呼び出せる「ツール」になる。
  docstringがツールの説明文としてLLMに渡され、LLMはこの説明を参考にツールを選ぶ。
"""

import math
import os

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from rag.llm import build_base_llm
from rag.retriever import retrieve

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


def _build_tools(vectorstore: Chroma) -> list:
    """エージェントが使用できるツールのリストを生成して返す。

    各ツールは @tool デコレータで定義する。
    docstring が LLM に渡されるツールの説明文になるため、
    「いつ使うか」を明確に記述することが重要。
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
        # DuckDuckGoSearchRunはAPIキー不要で使えるWeb検索ツール
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().run(query)

    @tool
    def calculator(expression: str) -> str:
        """数式を計算する。四則演算や数学関数が使える（例: '2 + 3 * 4', 'sqrt(16)', 'pi * 2'）。"""
        # math モジュールの関数のみ許可し、組み込み関数（__builtins__）を無効化
        # これにより eval() の悪用（ファイル削除等）を防ぐ
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


def build_agent(vectorstore: Chroma) -> object:
    """ReActエージェントを構築して返す。

    build_base_llm() を使う理由:
      create_react_agent は内部で llm.bind_tools(tools) を呼び出す。
      build_llm() が返す RunnableRetry ラッパーは bind_tools() に非対応のため、
      ラッパーなしの生のLLMインスタンスが必要。
    """
    llm = build_base_llm()
    tools = _build_tools(vectorstore)
    return create_react_agent(llm, tools)


def _print_tool_usage(messages: list) -> None:
    """エージェントが使用したツール呼び出しと結果をデバッグ出力する。

    --debug フラグ指定時のみ呼ばれる。
    AIMessage: LLMがツール呼び出しを決定したメッセージ
    ToolMessage: ツール実行結果のメッセージ
    """
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                # tool_callsは{"name": str, "args": dict, "id": str}の辞書
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                print(f"  [ツール呼び出し] {tc['name']}({args_str})")
        elif isinstance(msg, ToolMessage):
            # 長い結果は80文字でトリミングして表示
            preview = str(msg.content)[:80].replace("\n", " ")
            print(f"  [ツール結果] {preview}...")


def run_agent(
    vectorstore: Chroma,
    question: str,
    history: list[tuple[str, str]] | None = None,
    debug: bool = False,
) -> str:
    """エージェントを実行して回答を返す。

    会話履歴を HumanMessage / AIMessage のリストに変換してエージェントに渡すことで、
    前の会話文脈を踏まえた回答が可能になる（マルチターン対話）。

    Args:
        vectorstore: PDF検索に使用するChromaDBインスタンス
        question: ユーザーの質問文
        history: 過去の会話履歴 [(質問, 回答), ...]
        debug: True の場合ツール呼び出し履歴をコンソールに出力する
    """
    agent = build_agent(vectorstore)

    # 会話履歴をLangChainのMessageオブジェクトに変換
    messages: list = []
    for q, a in (history or []):
        messages.extend([HumanMessage(content=q), AIMessage(content=a)])
    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    if debug:
        _print_tool_usage(result["messages"])

    content = result["messages"][-1].content

    # AnthropicモデルはcontentをTextBlockのリスト形式で返す場合がある
    # 例: [{"type": "text", "text": "回答文", ...}, ...]
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content
