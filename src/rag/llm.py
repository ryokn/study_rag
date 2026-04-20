"""LLMファクトリ（Gemini / Ollama 切り替え対応）"""

import os

from langchain_core.language_models import BaseChatModel

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")


def build_base_llm() -> BaseChatModel:
    """生のLLMを返す（RAGAS等with_retry非対応ライブラリ向け）"""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def build_llm() -> BaseChatModel:
    """LLMを返す。Geminiの場合は429エラー時に自動リトライする。"""
    if LLM_PROVIDER == "ollama":
        # ローカル実行のためリトライ不要
        return build_base_llm()

    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
    return build_base_llm().with_retry(
        retry_if_exception_type=(ChatGoogleGenerativeAIError, Exception),
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )
