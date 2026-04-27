# SPEC.md — study-rag

## 概要

ローカルPDFファイルを対象としたRAG（Retrieval-Augmented Generation）サンプル実装。  
LangChain / LangGraph / Gemini API または Ollama を組み合わせた検索・回答生成パイプラインを学習目的で構築する。  
MLflowで実験を継続的に評価・比較できる仕組みを持つ。

---

## 技術スタック

| 役割 | 技術 |
|---|---|
| RAGフレームワーク | LangChain |
| エージェント/フロー制御 | LangGraph |
| LLM | Gemini API（`gemini-2.5-flash`）/ Azure OpenAI（`gpt-4o`等）/ Ollama（`gemma4`等） |
| Embedding | Gemini API（`gemini-embedding-001`）/ Azure OpenAI（`text-embedding-3-small`等） |
| ベクターDB | ChromaDB（ローカル永続化） |
| 実験管理 | MLflow |
| Web UI | Streamlit |
| 言語 | Python 3.13 |
| パッケージ管理 | uv |

---

## 機能要件

### フェーズ1: CLIでのRAG基本動作

- [x] PDFファイルの読み込みとテキスト抽出
- [x] テキストのチャンク分割
- [x] Gemini Embeddingでベクトル化してChromaDBに保存
- [x] ユーザー質問をベクトル検索して関連チャンクを取得
- [x] LLMで回答生成
- [x] CLIから質問→回答のループ実行

### フェーズ2: LangGraphによるフロー制御

- [x] 検索→判定→再検索 のグラフフロー実装
- [x] 回答品質が低い場合の再試行ロジック

### フェーズ3: MLflowによる実験評価

- [x] 各実験（チャンクサイズ、top-k）をMLflowにログ
- [x] 回答の品質スコア（faithfulness / answer_relevancy）を記録
- [x] 実験結果の比較・可視化（SQLiteバックエンド）
- [x] JSONファイルによる評価データセット管理（`--questions-file` オプション）

### フェーズ4: Streamlit Web UI

- [x] PDFアップロード画面
- [x] チャット形式の質問・回答UI

### フェーズ5: Ollama対応（ローカルLLM）

- [x] `LLM_PROVIDER` 環境変数で Gemini / Ollama を切り替え
- [x] `build_base_llm()` / `build_llm()` の分離（RAGAS・エージェント互換）

### フェーズ6: 会話履歴対応（マルチターン）

- [x] `RAGState` に `history` フィールドを追加
- [x] プロンプトに過去の会話コンテキストを含める
- [x] CLI・Web UI 両対応

### フェーズ7: Agentic AI（ReActエージェント）

- [x] LangGraph の ReAct エージェントを実装（`src/rag/agent.py`）
- [x] ツール: PDF検索・DuckDuckGo Web検索・計算・Pythonコード実行
- [x] `chat --agent` フラグ / Web UI トグルで切り替え
- [x] `--debug` フラグでツール使用履歴を表示
- [x] Gemini / Ollama 両対応

### フェーズ8: DuckDBによる構造化データ検索

- [x] DuckDB でCSVファイルをインメモリテーブルとして読み込む（`src/rag/table_search.py`）
- [x] LLMが自然言語→DuckDB SQLに自動変換して実行
- [x] `table` サブコマンド追加（CLI）
- [x] Web UI にテーブル検索モード追加（サイドバーのラジオボタンで切り替え）
- [x] サンプルCSV自動生成（都道府県人口・商品カタログ）

### フェーズ9: Azure OpenAI対応（マルチプロバイダー）

- [x] `LLM_PROVIDER=azure_openai` で Azure OpenAI（GPT-4o等）に切り替え可能
- [x] `AzureChatOpenAI` / `AzureOpenAIEmbeddings` を `langchain-openai` で実装
- [x] `build_embeddings()` を `llm.py` に新設しEmbedding生成を一元管理
- [x] `ingest.py` / `evaluator.py` の Embedding 生成を `build_embeddings()` に統一
- [x] Gemini / Azure OpenAI / Ollama の3プロバイダーに対応

### フェーズ10: 表・図を含むPDFの高精度取り込み（PyMuPDF4LLM）

- [x] `pymupdf4llm` でPDFをMarkdown形式に変換し表構造を保持
- [x] 環境変数 `PDF_LOADER` で `pypdf`（既存）/ `pymupdf4llm` を切り替え可能
- [x] `ingest` 実行時にチャンク内容をJSONログとして出力（`CHUNK_LOG_DIR` 環境変数）

### フェーズ11: マルチエージェント（Supervisor パターン）

- [x] LangGraph の Supervisor パターンで3エージェント構成を実装（`src/rag/multi_agent.py`）
- [x] Supervisor → ResearchAgent（PDF・Web検索）→ AnswerAgent（回答生成）のフロー
- [x] `multi-agent` サブコマンドを追加（`--debug` フラグでエージェント動作を表示）
- [x] Human in the Loop（HITL）対応: ResearchAgent 完了後にユーザーが確認・追加指示を入力可能
- [x] `MemorySaver` チェックポインターで `interrupt()` による一時停止・再開を実装

---

## 処理フロー

### 1. 取り込みフロー（`ingest` コマンド）

```
data/pdfs/*.pdf
    │
    ▼ PyPDFLoader（rag/ingest.py）
ページ単位のDocumentリスト
    │
    ▼ RecursiveCharacterTextSplitter（CHUNK_SIZE=500 / CHUNK_OVERLAP=50）
チャンクリスト
    │
    ▼ GoogleGenerativeAIEmbeddings（gemini-embedding-001）
    │
    ▼ Chroma.from_documents / add_documents
ChromaDB（./chroma_db/ に永続保存）
```

---

### 2. 質問応答フロー（`chat` コマンド / Web UI）

**RAGモード**（デフォルト）: LangGraph が以下のノードグラフを制御する。

```
ユーザーの質問
    │
    ▼ 【search ノード】（rag/retriever.py）
    │  ChromaDB 類似検索（TOP_K=5件）
    │
    ▼ 【generate ノード】（rag/chain.py）
    │  コンテキスト＋質問 → LLM（Gemini or Ollama）→ 回答生成
    │
    ▼ 【judge ノード】（rag/graph.py）
    │  「回答は十分か？」をLLMに判定させる
    │  retry_count をインクリメント
    │
    ▼ 【should_retry 分岐】
    ├─ 十分 or retry_count >= 2 ──→ 回答を返す
    └─ 不十分 ──────────────────→ search に戻る（最大2回再試行）
```

**エージェントモード**（`--agent` フラグ / Web UI トグル）: LangGraph の ReAct エージェントが以下のツールを自律的に選択・実行する。

```
ユーザーの質問
    │
    ▼ 【ReAct エージェント】（rag/agent.py）
    │  利用可能なツール:
    │  ├─ search_pdf    : ChromaDB でPDF検索
    │  ├─ web_search    : DuckDuckGo でWeb検索
    │  ├─ calculator    : 数式・数学関数の計算
    │  └─ python_repl   : Pythonコード実行
    │
    ▼ ツール呼び出し → 結果を観察 → 必要なら再度ツール選択
    │
    └─ 十分な情報が揃ったら回答を返す
```

**テーブル検索モード**（`table` コマンド / Web UI テーブル検索）: LLMがCSVスキーマを参照してSQLを生成し、DuckDBで実行する。

```
ユーザーの自然言語質問
    │
    ▼ 【スキーマ取得】（rag/table_search.py）
    │  data/csv/*.csv → DuckDB インメモリテーブル
    │  全テーブルのカラム定義・行数を取得
    │
    ▼ 【NL→SQL変換】
    │  LLM（Gemini or Ollama）がスキーマを参照してSQLを生成
    │
    ▼ 【SQL実行】
    │  DuckDB でSQL実行 → DataFrameとして取得
    │
    └─ 結果 + 実行SQLを返す
```

**マルチエージェントモード**（`multi-agent` コマンド）: LangGraph の Supervisor パターンで複数エージェントが役割分担する。HITL により ResearchAgent 完了後にユーザーが確認できる。

```
ユーザーの質問
    │
    ▼ 【Supervisor ノード】（rag/multi_agent.py）
    │  質問を分析し次の担当エージェントを決定
    │
    ├─ 調査が必要 → 【ResearchAgent】（ReAct: PDF検索・Web検索）
    │                    │
    │                    ▼ [HITL] interrupt(): 調査結果をユーザーに提示
    │                    │  y=進む / r=追加調査の指示 / n=キャンセル
    │                    │
    │                    └─ Supervisor へ戻る（MemorySaver で状態を保持）
    │
    └─ 情報が揃った → 【AnswerAgent】（調査結果をもとに最終回答生成）
                            │
                            └─ 終了
```

---

### 3. 評価フロー（`eval` コマンド）

```
data/eval_questions.json（質問・正解リスト）
    │
    ▼ 質問ごとに質問応答フローを実行
回答リスト ＋ 取得チャンクリスト
    │
    ▼ RAGAS（rag/evaluator.py）
    │  ├─ faithfulness       : 回答がコンテキストに忠実か
    │  └─ answer_relevancy   : 回答が質問に関連しているか
    │
    ▼ MLflow（mlflow_tracking/experiments.py）
パラメータ＋スコアを mlruns.db に記録
    │
    ▼ mlflow ui --backend-store-uri sqlite:///mlruns.db（http://localhost:5000）
実験結果の比較・可視化
```

---

## システム構成

```
study_rag/
├── pyproject.toml
├── .env.example
│
├── src/
│   ├── main.py              # CLIエントリーポイント
│   ├── app.py               # Streamlitエントリーポイント
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py        # PDF読み込み・チャンク分割・DB保存
│       ├── retriever.py     # ChromaDBからの検索
│       ├── chain.py         # LangChainのRAGチェーン
│       ├── graph.py         # LangGraphのフロー定義（RAGモード）
│       ├── agent.py         # ReActエージェント（エージェントモード）
│       ├── multi_agent.py   # Supervisorパターンのマルチエージェント（HITL対応）
│       ├── table_search.py  # DuckDBによるCSV構造化データ検索
│       ├── evaluator.py     # 回答品質評価
│       └── llm.py           # LLMファクトリ（Gemini / Azure OpenAI / Ollama 切り替え）
│   └── mlflow_tracking/
│       └── experiments.py   # MLflow実験ログユーティリティ
│
├── docs/
│   ├── SPEC.md
│   └── ENHANCEMENTS.md
│
├── data/
│   ├── pdfs/                         # 入力PDFの置き場所（gitignore対象）
│   └── eval_questions.example.json   # 評価データのサンプルテンプレート
│
└── chroma_db/               # ChromaDB永続化ディレクトリ（gitignore対象）
```

---

## 設定値（調整対象パラメータ）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | チャンク分割サイズ（文字数） |
| `CHUNK_OVERLAP` | 50 | チャンク間のオーバーラップ |
| `TOP_K` | 5 | 検索で取得するチャンク数 |
| `LLM_PROVIDER` | `gemini` | LLMプロバイダー（`gemini` / `azure_openai` / `ollama`） |
| `LLM_MODEL` | `gemini-2.5-flash` | チャット用モデル名またはデプロイメント名（Azure例: `gpt-4o`） |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | 埋め込みモデル（Gemini使用時） |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-small` | Azure Embedding デプロイメント名 |
| `PDF_LOADER` | `pypdf` | PDFローダー（`pypdf` / `pymupdf4llm`） |
| `CHUNK_LOG_DIR` | `./logs` | チャンクJSONログ出力先 |

これらの値をMLflowで変えながら実験比較する。

---

## 評価データフォーマット

評価データは `data/eval_questions.example.json` をコピーして作成する。

```json
[
  {
    "question": "質問テキスト",
    "ground_truth": "正解テキスト（任意）"
  }
]
```

- `ground_truth` は省略可能。省略時は `faithfulness` / `answer_relevancy` のみ計算される
- `ground_truth` を含めると正解との一致度も評価対象になる
- 5〜20件を目安に用意する（多いほどMLflowの比較精度が上がる）

---

<!-- AUTO-GENERATED -->
## 環境変数

`.env.example` より生成。

### 共通

| 変数名 | 必須 | 説明 | 例 |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | No | MLflow バックエンドURI | `sqlite:///mlruns.db` |
| `CHROMA_PERSIST_DIR` | No | ChromaDB 永続化ディレクトリ | `./chroma_db` |
| `LLM_PROVIDER` | No | LLMプロバイダー（デフォルト: `gemini`） | `gemini` / `azure_openai` / `ollama` |
| `EMBED_BATCH_SIZE` | No | Embedding バッチサイズ（デフォルト: `80`） | `500` |
| `EMBED_BATCH_INTERVAL` | No | バッチ間待機秒数（デフォルト: `65`） | `0` |
| `LLM_REQUEST_INTERVAL` | No | LLMリクエスト間隔（秒, デフォルト: `13`） | `2` |
| `PDF_LOADER` | No | PDFローダー切り替え（デフォルト: `pypdf`） | `pypdf` / `pymupdf4llm` |
| `CHUNK_LOG_DIR` | No | チャンクJSONログ出力先 | `./logs` |

### Gemini（`LLM_PROVIDER=gemini`）

| 変数名 | 必須 | 説明 | 例 |
|---|---|---|---|
| `GOOGLE_API_KEY` | **Yes** | Gemini API キー | `AIza...` |
| `LLM_MODEL` | No | チャット用モデル（デフォルト: `gemini-2.0-flash-lite`） | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | No | Embedding モデル | `gemini-embedding-001` |

### Azure OpenAI（`LLM_PROVIDER=azure_openai`）

| 変数名 | 必須 | 説明 | 例 |
|---|---|---|---|
| `AZURE_OPENAI_API_KEY` | **Yes** | Azure OpenAI APIキー | `sk-...` |
| `AZURE_OPENAI_ENDPOINT` | **Yes** | Azure リソースエンドポイント | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | No | API バージョン | `2024-02-01` |
| `LLM_MODEL` | **Yes** | チャット用デプロイメント名 | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | **Yes** | Embedding デプロイメント名 | `text-embedding-3-small` |

### Ollama（`LLM_PROVIDER=ollama`）

| 変数名 | 必須 | 説明 | 例 |
|---|---|---|---|
| `LLM_MODEL` | **Yes** | チャット用モデル名 | `gemma4:e2b` |
| `OLLAMA_EMBEDDING_MODEL` | **Yes** | Embedding モデル名 | `nomic-embed-text` |
<!-- AUTO-GENERATED -->

---

## 非機能要件

- ローカル完結（ChromaDB・MLflowはすべてローカル）
- `.env` ファイルでAPIキーを管理（`.gitignore` 対象）
- `data/pdfs/` と `chroma_db/` は `.gitignore` 対象
- 各フェーズは独立して動作確認できること

---

## 実装済み機能

1. `src/rag/ingest.py` — PDF読み込み・ChromaDB保存（PyPDF / PyMuPDF4LLM 切り替え対応）
2. `src/rag/retriever.py` + `src/rag/chain.py` — CLIでの検索・回答
3. `src/rag/graph.py` — LangGraphフロー（検索→生成→判定→再試行）
4. `src/mlflow_tracking/` — 実験ログ・RAGAS評価
5. `src/app.py` — Streamlit Web UI
6. `src/rag/llm.py` — Gemini / Azure OpenAI / Ollama 3プロバイダー対応
7. 会話履歴（マルチターン）対応
8. `src/rag/agent.py` — ReActエージェント（`--agent` フラグ）
9. `src/rag/table_search.py` — DuckDB による CSV 構造化データ検索
10. `src/rag/multi_agent.py` — Supervisor パターンのマルチエージェント（HITL対応）
