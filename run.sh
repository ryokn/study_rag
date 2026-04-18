#!/usr/bin/env bash
set -euo pipefail

BOLD=$(tput bold 2>/dev/null || echo "")
RESET=$(tput sgr0 2>/dev/null || echo "")
CYAN=$(tput setaf 6 2>/dev/null || echo "")
YELLOW=$(tput setaf 3 2>/dev/null || echo "")

show_menu() {
    echo ""
    echo "${BOLD}${CYAN}===== study-rag =====${RESET}"
    echo ""
    echo "  1) ingest       PDFを取り込む"
    echo "  2) chat         RAGチャット"
    echo "  3) chat-agent   エージェントチャット"
    echo "  4) chat-debug   エージェントチャット（デバッグ出力）"
    echo "  5) web          Web UI を起動"
    echo "  6) eval         品質評価（RAGAS + MLflow）"
    echo "  7) mlflow       MLflow UI を起動"
    echo "  q) quit         終了"
    echo ""
}

run_choice() {
    local choice=$1
    case $choice in
        1) uv run src/main.py ingest ;;
        2) uv run src/main.py chat ;;
        3) uv run src/main.py chat --agent ;;
        4) uv run src/main.py chat --agent --debug ;;
        5) uv run streamlit run src/app.py ;;
        6)
            printf "${YELLOW}評価ファイルのパス（未入力でデフォルト: data/eval_questions.json）: ${RESET}"
            read -r questions_file
            questions_file=${questions_file:-data/eval_questions.json}

            printf "${YELLOW}実験名（未入力でスキップ）: ${RESET}"
            read -r run_name

            if [[ -n "$run_name" ]]; then
                uv run src/main.py eval --questions-file "$questions_file" --run-name "$run_name"
            else
                uv run src/main.py eval --questions-file "$questions_file"
            fi
            ;;
        7) uv run mlflow ui --backend-store-uri sqlite:///mlruns.db ;;
        q|Q) echo "終了します。"; exit 0 ;;
        *) echo "無効な選択です。1〜7 または q を入力してください。" ;;
    esac
}

# .envが存在しない場合は警告
if [[ ! -f .env ]]; then
    echo "${YELLOW}警告: .env ファイルが見つかりません。cp .env.example .env で作成してください。${RESET}"
fi

while true; do
    show_menu
    printf "選択 [1-7/q]: "
    read -r choice
    echo ""
    run_choice "$choice"
done
