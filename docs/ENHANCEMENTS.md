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

## 3. 会話履歴対応（マルチターン）✅

- 現状は質問ごとに独立した検索・回答だが、前の会話を踏まえた応答ができるようにする
- LangChain の ConversationBufferMemory または LangGraph の状態管理で履歴を保持
- Web UI（Streamlit）のチャット画面にも履歴表示を追加

**学べること**
- LangChain のメモリ管理（短期・長期）
- 会話コンテキストを考慮したRAGのプロンプト設計
- マルチターン対話の評価方法

---

## 4. マルチプロバイダー対応（Gemini / Azure OpenAI / Ollama）✅

- LLM・Embeddingを環境変数1つで切り替えられるマルチプロバイダー構成を実装
- `build_embeddings()` を新設し、Embedding 生成を `llm.py` に一元管理
- Ollama 使用時は `nomic-embed-text` 等の専用 Embedding モデルに対応

**学べること**
- LangChain の抽象化レイヤー（`BaseChatModel` / `Embeddings` インターフェース）
- プロバイダー切り替えのファクトリーパターン
- ローカル完結RAGの構築（Ollama + nomic-embed-text + ChromaDB）

---

## 5. 表・図を含むPDFの高精度取り込み（PyMuPDF4LLM）✅

- 現状の `PyPDFLoader` は表がスペース区切りの崩れたテキストになる問題がある
- `pymupdf4llm` を使いPDFをMarkdown形式に変換することで表構造を保持する
- 図のキャプション・段組レイアウトも正しく処理できるようになる
- `ingest` 実行時にチャンク内容をJSONログとして出力（`CHUNK_LOG_DIR` 環境変数）

**実装済み**
- `src/rag/ingest.py` に `_load_pdf_pymupdf4llm()` を追加
- 環境変数 `PDF_LOADER=pymupdf4llm` で切り替え可能

**学べること**
- PDFパーサーの違いと使い分け（テキスト抽出 vs Markdown変換）
- Markdown形式チャンクがRAG精度に与える影響
- 表・図を含む技術文書のRAG設計

---

## 6. マルチエージェント（Supervisor パターン + Human in the Loop）✅

- 単一 ReAct エージェントを役割分担した複数エージェント構成に拡張
- Supervisor が質問を分析し ResearchAgent（情報収集）→ AnswerAgent（回答生成）へ振り分ける
- Human in the Loop（HITL）: ResearchAgent 完了後にユーザーが調査結果を確認して続行・追加指示・キャンセルを選択できる

**実装済み**
- `src/rag/multi_agent.py`: LangGraph Supervisor パターン + `interrupt()` / `MemorySaver`
- `multi-agent` サブコマンド（`--debug` フラグでエージェント動作を表示）
- HITL操作: `y`=回答生成へ進む / 任意テキスト=追加調査指示 / `n`=キャンセル

**学べること**
- LangGraph のマルチエージェントアーキテクチャ（Supervisor パターン）
- エージェント間の状態共有（TypedDict / StateGraph）と通信設計
- Human in the Loop の実装（`interrupt()` / `MemorySaver` / `Command(resume=...)`）
