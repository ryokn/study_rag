"""MLflowを使った実験ログユーティリティ"""

import os
import mlflow

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "study-rag")


def log_experiment(
    params: dict,
    metrics: dict,
    run_name: str | None = None,
) -> None:
    """
    パラメータと評価指標をMLflowに記録する。

    Args:
        params: チャンクサイズ・top-k等のハイパーパラメータ
        metrics: faithfulness・answer_relevancy等のスコア
        run_name: 実験の識別名
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
