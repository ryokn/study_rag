"""MLflowを使った実験ログユーティリティ

MLflowとは:
  機械学習実験の管理・比較ツール。パラメータ（設定値）とメトリクス（評価スコア）を
  記録することで、「チャンクサイズを変えたら品質スコアはどう変わるか？」といった
  実験比較を可視化UIで確認できる。

実験の確認方法:
  uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
  → http://localhost:5000 でブラウザから確認
"""

import os

import mlflow

# 実験グループ名（MLflow UI上でのグループ名）
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "study-rag")


def log_experiment(
    params: dict,
    metrics: dict,
    run_name: str | None = None,
) -> None:
    """パラメータと評価指標を1回の実験（Run）としてMLflowに記録する。

    Args:
        params: ハイパーパラメータの辞書
                例: {"chunk_size": "500", "top_k": "5", "llm_model": "gemini-2.5-flash"}
        metrics: 評価スコアの辞書
                例: {"faithfulness": 0.85, "answer_relevancy": 0.92}
        run_name: 実験の識別名（省略時は自動生成）
    """
    # トラッキングURIはSQLiteファイルに保存（デフォルト: ./mlruns.db）
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))

    # 実験グループを設定（存在しない場合は自動作成）
    mlflow.set_experiment(EXPERIMENT_NAME)

    # with ブロック内の記録が1つの「Run」としてまとめられる
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)   # 設定値を記録
        mlflow.log_metrics(metrics) # スコアを記録
