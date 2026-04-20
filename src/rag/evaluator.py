"""RAGASを使った回答品質評価

RAGASとは:
  RAG（Retrieval-Augmented Generation）システムの品質を自動評価するフレームワーク。
  LLMを使って評価を行うため、人手による評価なしで品質スコアを算出できる。

評価指標:
  - faithfulness（忠実性）:
      回答がコンテキスト（検索結果）の内容に基づいているか。
      コンテキストに書かれていないことを「作り話」していないかを測る（0〜1）。
  - answer_relevancy（回答関連性）:
      回答が質問に正しく答えているか。的外れな回答でないかを測る（0〜1）。

スコアが高いほど品質が良く、1.0が満点。
"""

import os

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from rag.llm import build_base_llm

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# 無料枠対応: 並列リクエスト数を1に絞り、タイムアウトを長めに設定
# max_workers=1 はレート制限（1分あたりのリクエスト数上限）を超えないための設定
_RUN_CONFIG = RunConfig(max_workers=1, timeout=180)


def _build_ragas_llm() -> LangchainLLMWrapper:
    """RAGASが使用するLLMラッパーを返す。

    build_base_llm() を使う理由:
      RAGASは内部で agenerate_prompt() を呼び出す。
      build_llm() の RunnableRetry ラッパーはこのメソッドに非対応のため、
      生のLLMインスタンスを LangchainLLMWrapper で包む必要がある。
    """
    return LangchainLLMWrapper(build_base_llm())


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """RAGASが使用するEmbeddingラッパーを返す。

    answer_relevancy の計算には質問と回答のベクトル類似度が使われるため
    Embeddingモデルも必要になる。
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, float]:
    """RAGASで回答品質を評価してスコアを返す。

    Args:
        questions: 評価する質問のリスト
        answers: RAGシステムが生成した回答のリスト
        contexts: 各質問に対して取得したチャンクのリスト（二重リスト）
        ground_truths: 正解回答のリスト（任意）。指定すると追加指標も計算される

    Returns:
        {"faithfulness": float, "answer_relevancy": float} 形式のスコア辞書
    """
    # RAGASはHuggingFace Datasetsの形式でデータを受け取る
    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    # 各メトリクスに使用するLLM/Embeddingを明示的に設定する
    # （RAGASのグローバル設定を上書きすることでGeminiを使用させる）
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        run_config=_RUN_CONFIG,
    )
    # 全サンプルのスコアを平均して返す
    return result.to_pandas().mean(numeric_only=True).to_dict()
