"""LLMファクトリ（Gemini / Ollama 切り替え対応）

このモジュールはLLMインスタンスの生成を一元管理する。
2種類のファクトリ関数を用意している理由:
  - build_base_llm(): 生のLLMを返す。RagasやLangGraphエージェントなど
                       `bind_tools()` や `agenerate_prompt()` を直接呼ぶ
                       ライブラリに対して使用する。
  - build_llm()     : Gemini向けに429(レート超過)エラー時の自動リトライを
                       ラップした版。通常のRAGチェーン・グラフで使用する。
"""

import os

from langchain_core.language_models import BaseChatModel

# 環境変数でプロバイダとモデルを切り替える（.envで設定）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def build_base_llm() -> BaseChatModel:
    """生のLLMインスタンスを返す。

    RAGAS・エージェント（bind_tools必須）など、
    with_retry() ラッパーが干渉するケースで使用する。
    temperature=0 は回答の再現性を高めるために設定（0=確定的、1=ランダム）。
    """
    if LLM_PROVIDER == "ollama":
        # Ollamaはローカル実行のためlangchain-ollamaパッケージを使用
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, temperature=0)

    # デフォルトはGemini API（Google AI Studio / Vertex AI）
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def build_llm() -> BaseChatModel:
    """リトライ付きLLMインスタンスを返す。

    Geminiの無料枠は1分あたりのリクエスト数に上限がある。
    with_retry() によって429エラー（レート超過）時に指数バックオフで
    自動リトライするため、通常のRAGチェーンで安定して動作する。

    Ollamaはローカル実行なのでリトライ不要、base_llmをそのまま返す。
    """
    if LLM_PROVIDER == "ollama":
        return build_base_llm()

    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    # with_retry()はRunnableをラップする LangChain の仕組み
    # stop_after_attempt=5: 最大5回まで再試行
    # wait_exponential_jitter=True: 再試行間隔をランダムにずらして競合を防ぐ
    return build_base_llm().with_retry(
        retry_if_exception_type=(ChatGoogleGenerativeAIError, Exception),
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )
