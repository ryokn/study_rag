"""RAGASを使った回答品質評価"""

import os

from datasets import Dataset
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from rag.llm import build_base_llm

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

# 無料枠対応: 並列数1・タイムアウト長めに設定
_RUN_CONFIG = RunConfig(max_workers=1, timeout=180)


def _build_ragas_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(build_base_llm())


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(embeddings)


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, float]:
    """
    RAGASで回答品質を評価する。

    Returns:
        {"faithfulness": float, "answer_relevancy": float}
    """
    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    # 各メトリクスにGeminiのLLM/Embeddingを明示的に設定
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        run_config=_RUN_CONFIG,
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()
