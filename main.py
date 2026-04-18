"""CLIエントリーポイント"""

import argparse
import os
import time

from dotenv import load_dotenv

load_dotenv()


def cmd_ingest(args: argparse.Namespace) -> None:
    from rag.ingest import ingest
    ingest(args.pdf_dir)


def cmd_chat(args: argparse.Namespace) -> None:
    from rag.ingest import load_vectorstore
    from rag.graph import run_graph

    vectorstore = load_vectorstore()
    print("RAGチャット開始 (終了するには 'exit' を入力)\n")

    while True:
        question = input("質問: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue
        answer = run_graph(vectorstore, question)
        print(f"\n回答: {answer}\n")


def _load_eval_dataset(args: argparse.Namespace) -> tuple[list[str], list[str] | None]:
    """--questions-file または --questions から質問と正解リストを返す"""
    import json

    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            data = json.load(f)
        questions = [item["question"] for item in data]
        ground_truths = [item.get("ground_truth") for item in data]
        has_gt = all(gt is not None for gt in ground_truths)
        return questions, (ground_truths if has_gt else None)

    if args.questions:
        return [q.strip() for q in args.questions.split(",")], None

    raise ValueError("--questions または --questions-file のいずれかを指定してください")


def cmd_eval(args: argparse.Namespace) -> None:
    from rag.ingest import load_vectorstore
    from rag.retriever import retrieve
    from rag.chain import build_chain
    from rag.evaluator import evaluate_rag
    from mlflow_tracking.experiments import log_experiment

    # リクエスト間隔（LLMのリトライはrag/llm.pyのwith_retry()で処理）
    llm_request_interval = int(os.getenv("LLM_REQUEST_INTERVAL", "13"))

    questions, ground_truths = _load_eval_dataset(args)
    print(f"評価データ: {len(questions)} 件")

    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)

    answers: list[str] = []
    contexts: list[list[str]] = []

    for i, q in enumerate(questions):
        print(f"  質問 {i + 1}/{len(questions)}: {q[:30]}...")
        docs = retrieve(vectorstore, q)
        answers.append(chain.invoke(q))
        contexts.append([doc.page_content for doc in docs])
        if i < len(questions) - 1:
            time.sleep(llm_request_interval)

    metrics = evaluate_rag(questions, answers, contexts, ground_truths)
    print("評価結果:", metrics)

    params = {
        "chunk_size": os.getenv("CHUNK_SIZE", "500"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP", "50"),
        "top_k": os.getenv("TOP_K", "5"),
        "llm_model": os.getenv("LLM_MODEL", "gemini-2.0-flash-lite"),
        "eval_size": str(len(questions)),
    }
    log_experiment(params=params, metrics=metrics, run_name=args.run_name)
    print("MLflowに記録しました")


def main() -> None:
    parser = argparse.ArgumentParser(description="study-rag CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_ingest = subparsers.add_parser("ingest", help="PDFをChromaDBに取り込む")
    p_ingest.add_argument("--pdf-dir", default="./data/pdfs", help="PDFディレクトリ")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chat = subparsers.add_parser("chat", help="RAGチャットを開始する")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = subparsers.add_parser("eval", help="RAGASで品質評価しMLflowに記録する")
    p_eval_group = p_eval.add_mutually_exclusive_group()
    p_eval_group.add_argument("--questions-file", default=None, help="評価データJSONファイルのパス")
    p_eval_group.add_argument("--questions", default=None, help="カンマ区切りの質問リスト（簡易実行用）")
    p_eval.add_argument("--run-name", default=None, help="MLflow実験名")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
