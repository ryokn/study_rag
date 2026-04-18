"""リトライ付きGemini LLMファクトリ"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def build_base_llm() -> ChatGoogleGenerativeAI:
    """生のGemini LLMを返す（RAGAS等with_retry非対応ライブラリ向け）"""
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def build_llm() -> ChatGoogleGenerativeAI:
    """429エラー時に自動リトライするLLMを返す。

    LangChainのwith_retry()を使い、429 / 5xx発生時に
    指数バックオフで最大5回リトライする。
    """
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    return build_base_llm().with_retry(
        retry_if_exception_type=(ChatGoogleGenerativeAIError, Exception),
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )
