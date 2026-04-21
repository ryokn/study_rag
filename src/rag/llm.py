"""LLMファクトリ（Gemini / Ollama 切り替え対応）

このモジュールはLLMインスタンスとEmbeddingインスタンスの生成を一元管理する。

関数の使い分け:
  - build_base_llm() : 生のLLMを返す。RAGASやLangGraphエージェントなど
                       `bind_tools()` や `agenerate_prompt()` を直接呼ぶ
                       ライブラリに対して使用する。
  - build_llm()      : Gemini向けにレート制限エラー時の自動リトライをラップした版。
                       通常のRAGチェーン・グラフで使用する。
  - build_embeddings(): プロバイダーに合わせたEmbeddingインスタンスを返す。
                       ChromaDB保存・RAGAS評価で使用する。

環境変数:
  LLM_PROVIDER           : gemini または ollama（デフォルト: gemini）
  LLM_MODEL              : チャット用モデル名（例: gemini-2.5-flash / gemma4:2b）
  EMBEDDING_MODEL        : Gemini Embeddingモデル名（gemini使用時）
  OLLAMA_EMBEDDING_MODEL : Ollama Embeddingモデル名（ollama使用時）
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# 環境変数でプロバイダとモデルを切り替える（.envで設定）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Gemini Embedding モデル名
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Ollama Embedding モデル名（チャットモデルとは別に専用モデルを使う）
# nomic-embed-text は軽量で日本語もある程度対応している
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


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


def build_embeddings() -> Embeddings:
    """プロバイダーに対応したEmbeddingインスタンスを返す。

    ChromaDBへの保存（ingest.py）とRAGAS評価（evaluator.py）の両方から
    呼び出されるため、このモジュールで一元管理する。

    重要: ingest と chat で必ず同じモデルが返されるため、
    モデルを変更した場合は chroma_db/ を削除して再ingestが必要。

    Returns:
        Embeddings: LangChain の Embeddings インターフェースを実装したインスタンス
    """
    if LLM_PROVIDER == "ollama":
        # OllamaEmbeddings はチャットモデル（LLM_MODEL）とは別の
        # 専用 Embedding モデルを使う（OLLAMA_EMBEDDING_MODEL で指定）
        # 事前に `ollama pull nomic-embed-text` が必要
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    # デフォルトは Gemini Embedding API
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
