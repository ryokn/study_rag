"""RAGASを使った回答品質評価"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy


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
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy]
    result = evaluate(dataset, metrics=metrics)
    return result.to_pandas().mean(numeric_only=True).to_dict()
