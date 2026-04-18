# study-rag

LangChain / LangGraph / Gemini API を使ったRAGサンプル実装。  
ローカルPDFを対象に検索・回答生成・品質評価までを学習目的で構築したプロジェクト。

## 技術スタック

| 役割 | 技術 |
|---|---|
| RAGフレームワーク | LangChain |
| エージェント/フロー制御 | LangGraph |
| LLM / Embedding | Gemini API |
| ベクターDB | ChromaDB（ローカル永続化） |
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

`.env` を開いて `GOOGLE_API_KEY` に [Gemini API キー](https://aistudio.google.com/app/apikey) を設定してください。

```env
GOOGLE_API_KEY=your_gemini_api_key
```

**3. PDFファイルを配置**

```bash
mkdir -p data/pdfs
cp your_document.pdf data/pdfs/
```

## 使い方

### PDFの取り込み

```bash
uv run python main.py ingest
```

`data/pdfs/` 内のPDFをチャンク分割してChromaDBに保存します。

### CLIチャット

```bash
uv run python main.py chat
```

```
RAGチャット開始 (終了するには 'exit' を入力)

質問: このドキュメントの概要を教えてください
回答: ...
```

### Web UI

```bash
uv run streamlit run app.py
```

ブラウザで `http://localhost:8501` を開きます。サイドバーからPDFをアップロードし、チャット欄で質問できます。

### 品質評価（RAGAS + MLflow）

**方法1: JSONファイルで評価（推奨）**

```bash
# テンプレートをコピーして評価データを作成
cp data/eval_questions.example.json data/eval_questions.json
# eval_questions.json を編集して質問と正解を記入

uv run python main.py eval \
  --questions-file data/eval_questions.json \
  --run-name "実験1_chunk500"
```

**方法2: コマンドラインで簡易実行**

```bash
uv run python main.py eval \
  --questions "質問1,質問2,質問3" \
  --run-name "実験1_chunk500"
```

RAGASで `faithfulness` / `answer_relevancy` を計算し、MLflowに記録します。

#### 評価データJSONのフォーマット

```json
[
  {
    "question": "返品ポリシーを教えてください",
    "ground_truth": "購入から30日以内であれば返品可能です"
  },
  {
    "question": "送料はいくらですか",
    "ground_truth": "5000円以上の購入で送料無料です"
  }
]
```

- `ground_truth`（正解）は省略可能。省略時は `faithfulness` / `answer_relevancy` のみ評価
- 5〜20件を目安に用意する

#### パラメータを変えて実験比較する例

```bash
# 実験1: デフォルト設定
uv run python main.py eval --questions-file data/eval_questions.json --run-name "chunk500_topk5"

# 実験2: チャンクサイズを小さく
CHUNK_SIZE=200 uv run python main.py eval --questions-file data/eval_questions.json --run-name "chunk200_topk5"

# 実験3: 取得チャンク数を増やす
TOP_K=10 uv run python main.py eval --questions-file data/eval_questions.json --run-name "chunk500_topk10"
```

### MLflow UIで実験比較

```bash
uv run mlflow ui
```

`http://localhost:5000` でチャンクサイズやtop-kを変えた実験結果を比較できます。

## 処理フロー

### 取り込みフロー（`ingest`）

```
data/pdfs/*.pdf
    ↓ PyPDFLoader でテキスト抽出
    ↓ RecursiveCharacterTextSplitter でチャンク分割
    ↓ Gemini Embedding でベクトル化（80件ずつ・65秒待機）
    ↓
ChromaDB（./chroma_db/）に永続保存
```

### 質問応答フロー（`chat` / Web UI）

LangGraph が「検索 → 生成 → 品質判定 → 再試行」のグラフを制御します。

```
ユーザーの質問
    ↓ ChromaDB 類似検索（top_k=5）
    ↓ Gemini LLM で回答生成
    ↓ 回答品質を Gemini に判定させる
    ├─ 十分 or 2回試行済み → 回答を返す
    └─ 不十分 → 再検索（最大2回）
```

### 評価フロー（`eval`）

```
eval_questions.json（質問・正解リスト）
    ↓ 質問ごとに質問応答フローを実行
    ↓ RAGAS で品質スコアを計算
       ├─ faithfulness（ハルシネーション検出）
       └─ answer_relevancy（回答の関連性）
    ↓
MLflow に記録 → http://localhost:5000 で実験比較
```

## ディレクトリ構成

```
study_rag/
├── main.py                  # CLIエントリーポイント
├── app.py                   # Streamlit Web UI
├── pyproject.toml
├── .env.example
│
├── rag/
│   ├── ingest.py            # PDF読み込み・チャンク分割・DB保存
│   ├── retriever.py         # ChromaDBからの検索
│   ├── chain.py             # LangChain RAGチェーン
│   ├── graph.py             # LangGraph フロー（検索→生成→品質判定→再試行）
│   └── evaluator.py         # RAGAS品質評価
│
├── mlflow_tracking/
│   └── experiments.py       # MLflow実験ログ
│
└── data/
    ├── pdfs/                          # 入力PDF（gitignore対象）
    └── eval_questions.example.json    # 評価データのテンプレート
```

## 調整可能なパラメータ

`.env` で以下を変更して MLflow で実験比較できます。

| 変数 | デフォルト | 説明 |
|---|---|---|
| `CHUNK_SIZE` | 500 | チャンク分割サイズ（文字数） |
| `CHUNK_OVERLAP` | 50 | チャンク間のオーバーラップ |
| `TOP_K` | 5 | 検索で取得するチャンク数 |
| `LLM_MODEL` | gemini-2.5-flash | 使用するGeminiモデル |
| `EMBEDDING_MODEL` | gemini-embedding-001 | 埋め込みモデル |

## 実装フェーズ

- [x] フェーズ1: CLIでのRAG基本動作
- [x] フェーズ2: LangGraphによるフロー制御（再試行ロジック）
- [x] フェーズ3: MLflow + RAGASによる実験評価
- [x] フェーズ4: Streamlit Web UI
