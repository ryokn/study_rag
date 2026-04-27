# study-rag

LangChain / LangGraph / Gemini API・Azure OpenAI・Ollama を使ったRAGサンプル実装。  
ローカルPDFを対象に検索・回答生成・品質評価までを学習目的で構築したプロジェクト。

## 技術スタック

| 役割 | 技術 |
|---|---|
| RAGフレームワーク | LangChain |
| エージェント/フロー制御 | LangGraph |
| LLM / Embedding | Gemini API / Azure OpenAI / Ollama（ローカルLLM） |
| ベクターDB | ChromaDB（ローカル永続化） |
| 構造化データ検索 | DuckDB（インメモリSQL） |
| 実験管理 | MLflow |
| Web UI | Streamlit |
| パッケージ管理 | uv |

## セットアップ

**1. 依存関係のインストール**

```bash
uv sync
```

**2. 環境変数の設定**

```bash
cp .env.example .env
```

`.env` を開いて使用するプロバイダーに合わせて設定します。

**Gemini API を使う場合：**
```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-2.5-flash
```

**Azure OpenAI を使う場合：**
```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
LLM_MODEL=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

**Ollama（ローカルLLM）を使う場合：**
```bash
brew install ollama
ollama pull gemma4:e2b
ollama pull nomic-embed-text
ollama serve  # 別ターミナルで起動
```

```env
LLM_PROVIDER=ollama
LLM_MODEL=gemma4:e2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

**3. PDFファイルを配置**

```bash
mkdir -p data/pdfs
cp your_document.pdf data/pdfs/
```

## コマンドリファレンス

<!-- AUTO-GENERATED -->
| コマンド | 説明 |
|---|---|
| `uv run src/main.py ingest` | PDF を ChromaDB に取り込む |
| `uv run src/main.py ingest --pdf-dir ./data/pdfs` | 取り込みディレクトリを指定 |
| `uv run src/main.py chat` | RAGチャット（LangGraphフロー） |
| `uv run src/main.py chat --agent` | エージェントモード（PDF・Web検索・計算・コード実行） |
| `uv run src/main.py chat --agent --debug` | エージェントモード + ツール使用ログ表示 |
| `uv run src/main.py multi-agent` | マルチエージェント（Supervisor + ResearchAgent + AnswerAgent + HITL） |
| `uv run src/main.py multi-agent --debug` | マルチエージェント + エージェント動作ログ表示 |
| `uv run src/main.py table` | CSV を自然言語で検索（DuckDB NL→SQL） |
| `uv run src/main.py table --csv-dir ./data/csv` | 検索ディレクトリを指定 |
| `uv run src/main.py eval --questions-file data/eval_questions.json` | RAGAS評価 + MLflow記録 |
| `uv run src/main.py eval --questions "質問1,質問2"` | 簡易評価（JSONファイル不要） |
| `uv run streamlit run src/app.py` | Web UI を起動（http://localhost:8501） |
| `uv run mlflow ui --backend-store-uri sqlite:///mlruns.db` | MLflow UI を起動（http://localhost:5000） |
<!-- END AUTO-GENERATED -->

### multi-agent の HITL 操作

`multi-agent` は ResearchAgent が調査を完了するとユーザーに確認を求めます。

```
=== 調査結果プレビュー ===
（調査内容の抜粋）

[HITL] y=回答生成へ進む / r=追加調査の指示を入力 / n=キャンセル:
```

| 入力 | 動作 |
|---|---|
| `y` | AnswerAgent が最終回答を生成 |
| 任意のテキスト | そのテキストを指示として追加調査を実行 |
| `n` | キャンセル |

### PDFローダーの切り替え

表・図を含むPDFは `pymupdf4llm` で精度が向上します。

```env
PDF_LOADER=pymupdf4llm   # デフォルトは pypdf
```

## ディレクトリ構成

```
study_rag/
├── pyproject.toml
├── .env.example
├── TASKS.md                          # タスク管理
│
├── src/
│   ├── main.py                      # CLIエントリーポイント
│   ├── app.py                       # Streamlit Web UI
│   ├── rag/
│   │   ├── ingest.py                # PDF読み込み・チャンク分割・DB保存
│   │   ├── retriever.py             # ChromaDBからの検索
│   │   ├── chain.py                 # LangChain RAGチェーン
│   │   ├── graph.py                 # LangGraph フロー（RAGモード）
│   │   ├── agent.py                 # ReActエージェント
│   │   ├── multi_agent.py           # Supervisorパターン マルチエージェント（HITL対応）
│   │   ├── table_search.py          # DuckDB CSV検索（NL→SQL）
│   │   ├── evaluator.py             # RAGAS品質評価
│   │   └── llm.py                   # LLMファクトリ（3プロバイダー対応）
│   └── mlflow_tracking/
│       └── experiments.py           # MLflow実験ログ
│
├── docs/
│   ├── SPEC.md                      # 内部設計仕様
│   └── flow.md                      # Mermaidフロー図
│
└── data/
    ├── pdfs/                        # 入力PDF（gitignore対象）
    ├── csv/                         # 入力CSV（tableコマンド用）
    └── eval_questions.example.json  # 評価データのテンプレート
```

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [docs/SPEC.md](docs/SPEC.md) | 機能要件・処理フロー・環境変数リファレンス |
| [docs/flow.md](docs/flow.md) | Mermaidフロー図（全モード） |
| [TASKS.md](TASKS.md) | 今後の実装アイデア・タスク管理 |

## 実装フェーズ

- [x] フェーズ1: CLIでのRAG基本動作
- [x] フェーズ2: LangGraphによるフロー制御（再試行ロジック）
- [x] フェーズ3: MLflow + RAGASによる実験評価
- [x] フェーズ4: Streamlit Web UI
- [x] フェーズ5: Ollama対応（ローカルLLM切り替え）
- [x] フェーズ6: 会話履歴対応（マルチターン）
- [x] フェーズ7: Agentic AI（ReActエージェント + 複数ツール）
- [x] フェーズ8: DuckDBによる構造化データ検索（NL→SQL）
- [x] フェーズ9: Azure OpenAI対応（Gemini / Azure OpenAI / Ollama の切り替え）
- [x] フェーズ10: PyMuPDF4LLMによる高精度PDF取り込み（表・図対応）
- [x] フェーズ11: マルチエージェント（Supervisor パターン + HITL）
