"""LLMファクトリ（Gemini / Azure OpenAI / Ollama 切り替え対応）

このモジュールはLLMインスタンスとEmbeddingインスタンスの生成を一元管理する。

対応プロバイダー:
  - gemini      : Google Gemini API（デフォルト）
  - azure_openai: Azure OpenAI Service（GPT-4o等）
  - ollama      : ローカルLLM（Ollamaサーバーが必要）

関数の使い分け:
  - build_base_llm() : 生のLLMを返す。RAGASやLangGraphエージェントなど
                       `bind_tools()` や `agenerate_prompt()` を直接呼ぶ
                       ライブラリに対して使用する。
  - build_llm()      : レート制限エラー時の自動リトライをラップした版。
                       通常のRAGチェーン・グラフで使用する。
  - build_embeddings(): プロバイダーに合わせたEmbeddingインスタンスを返す。
                       ChromaDB保存・RAGAS評価で使用する。

環境変数:
  LLM_PROVIDER                    : gemini / azure_openai / ollama（デフォルト: gemini）
  LLM_MODEL                       : チャット用モデル名またはデプロイメント名
  EMBEDDING_MODEL                  : Gemini Embeddingモデル名（gemini使用時）
  OLLAMA_EMBEDDING_MODEL           : Ollama Embeddingモデル名（ollama使用時）
  AZURE_OPENAI_API_KEY             : Azure OpenAI APIキー（azure_openai使用時）
  AZURE_OPENAI_ENDPOINT            : Azure OpenAI エンドポイントURL
  AZURE_OPENAI_API_VERSION         : Azure OpenAI APIバージョン
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Azure Embedding用デプロイメント名

重要: ingest と chat で必ず同じ build_embeddings() の戻り値を使うこと。
      モデルを変更した場合は chroma_db/ を削除して再ingestが必要。
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# 環境変数でプロバイダとモデルを切り替える（.envで設定）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Gemini 専用の設定
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# Ollama Embedding モデル名（チャットモデルとは別に専用モデルを使う）
# nomic-embed-text は軽量で日本語もある程度対応している
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Azure OpenAI 専用の設定（azure_openai プロバイダー使用時のみ参照）
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)


def _build_azure_openai_llm() -> BaseChatModel:
    """Azure OpenAI の ChatLLM インスタンスを返す。

    AzureChatOpenAI は langchain-openai パッケージで提供される。
    azure_deployment にはAzureポータルで作成したデプロイメント名（LLM_MODEL）を指定する。
    temperature=0 は回答の再現性を高める設定（0=確定的、1=ランダム）。
    """
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=LLM_MODEL,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,          # type: ignore[arg-type]
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )


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

    if LLM_PROVIDER == "azure_openai":
        return _build_azure_openai_llm()

    # デフォルトはGemini API（Google AI Studio）
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def build_llm() -> BaseChatModel:
    """リトライ付きLLMインスタンスを返す。

    GeminiとAzure OpenAIはAPIレート制限があるため、429エラー時に
    指数バックオフで自動リトライするwith_retry()を適用する。
    Ollamaはローカル実行なのでリトライ不要。

    stop_after_attempt=5: 最大5回まで再試行
    wait_exponential_jitter=True: 再試行間隔をランダムにずらして競合を防ぐ
    """
    if LLM_PROVIDER == "ollama":
        return build_base_llm()

    return build_base_llm().with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=5,
    )


def build_embeddings() -> Embeddings:
    """プロバイダーに対応したEmbeddingインスタンスを返す。

    ChromaDBへの保存（ingest.py）とRAGAS評価（evaluator.py）の両方から
    呼び出されるため、このモジュールで一元管理する。

    Returns:
        Embeddings: LangChain の Embeddings インターフェースを実装したインスタンス
    """
    if LLM_PROVIDER == "ollama":
        # OllamaEmbeddings はチャットモデル（LLM_MODEL）とは別の
        # 専用 Embedding モデルを使う（OLLAMA_EMBEDDING_MODEL で指定）
        # 事前に `ollama pull nomic-embed-text` が必要
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    if LLM_PROVIDER == "azure_openai":
        # AzureOpenAIEmbeddings は langchain-openai パッケージで提供される
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,       # type: ignore[arg-type]
            api_version=AZURE_OPENAI_API_VERSION,
        )

    # デフォルトは Gemini Embedding API
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
