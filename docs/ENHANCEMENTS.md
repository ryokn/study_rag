# エンハンス計画メモ

学習目的で実装したい機能アイデア。

---

## 1. DuckDB を使った構造化データ検索 ✅

- 非構造化テキスト（PDF）だけでなく、CSVの構造化データも検索対象にする
- DuckDB でインメモリSQL検索、LLMが自然言語→SQLに自動変換
- `table` コマンド（CLI）/ テーブル検索モード（Web UI）で独立して利用可能
- `src/rag/table_search.py` に実装。`data/csv/` にCSVを置くだけで自動認識

**学べること**
- DuckDB の基本操作（`read_csv_auto`、インメモリDB）
- LLMを使ったNL→SQL変換パターン
- テキスト検索と構造化データ検索のハイブリッドRAGアーキテクチャ

---

## 2. Agentic AI の実装 ✅

- 単純な検索→回答だけでなく、複数ツールを自律的に選択・実行するエージェントを実装
- LangGraph の ReAct パターンで実装（`src/rag/agent.py`）
- ツール: PDF検索・DuckDuckGo Web検索・計算（math）・Pythonコード実行
- `chat --agent` フラグ / Web UI トグルで切り替え可能
- Gemini / Ollama 両対応

**学べること**
- LangGraph のマルチエージェント設計
- Tool use / Function calling の実装パターン
- エージェントの評価・デバッグ手法

---

## 3. 会話履歴対応（マルチターン）

- 現状は質問ごとに独立した検索・回答だが、前の会話を踏まえた応答ができるようにする
- LangChain の ConversationBufferMemory または LangGraph の状態管理で履歴を保持
- Web UI（Streamlit）のチャット画面にも履歴表示を追加

**学べること**
- LangChain のメモリ管理（短期・長期）
- 会話コンテキストを考慮したRAGのプロンプト設計
- マルチターン対話の評価方法
