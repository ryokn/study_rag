# SPEC.md — study-rag

## 概要

ローカルPDFファイルを対象としたRAG（Retrieval-Augmented Generation）サンプル実装。  
LangChain / LangGraph / Gemini API を組み合わせた検索・回答生成パイプラインを学習目的で構築する。  
MLflowで実験を継続的に評価・比較できる仕組みを持つ。

---

## 技術スタック

| 役割 | 技術 |
|---|---|
| RAGフレームワーク | LangChain |
| エージェント/フロー制御 | LangGraph |
| LLM / Embedding | Gemini API（`gemini-2.5-flash` / `gemini-embedding-001`） |
| ベクターDB | ChromaDB（ローカル永続化） |
| 実験管理 | MLflow |
| Web UI | Streamlit |
| 言語 | Python 3.13 |
| パッケージ管理 | uv |

---

## 機能要件

### フェーズ1: CLIでのRAG基本動作

- [ ] PDFファイルの読み込みとテキスト抽出
- [ ] テキストのチャンク分割
- [ ] Gemini Embeddingでベクトル化してChromaDBに保存
- [ ] ユーザー質問をベクトル検索して関連チャンクを取得
- [ ] Gemini LLMで回答生成
- [ ] CLIから質問→回答のループ実行

### フェーズ2: LangGraphによるフロー制御

- [ ] 検索→判定→再検索 のグラフフロー実装
- [ ] 回答品質が低い場合の再試行ロジック
- [ ] グラフの可視化

### フェーズ3: MLflowによる実験評価

- [ ] 各実験（チャンクサイズ、top-k、プロンプト）をMLflowにログ
- [ ] 回答の品質スコア（faithfulness / relevance）を記録
- [ ] 実験結果の比較・可視化
- [ ] JSONファイルによる評価データセット管理（`--questions-file` オプション）

### フェーズ4: Streamlit Web UI

- [ ] PDFアップロード画面
- [ ] チャット形式の質問・回答UI
- [ ] 参照チャンク（出典）の表示

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
    │  ※ 無料枠対応: 80件ずつ・65秒待機
    │
    ▼ Chroma.from_documents / add_documents
ChromaDB（./chroma_db/ に永続保存）
```

---

### 2. 質問応答フロー（`chat` コマンド / Web UI）

LangGraph が以下のノードグラフを制御する。

```
ユーザーの質問
    │
    ▼ 【search ノード】（rag/retriever.py）
    │  ChromaDB 類似検索（TOP_K=5件）
    │
    ▼ 【generate ノード】（rag/chain.py）
    │  コンテキスト＋質問 → Gemini LLM（gemini-2.5-flash）→ 回答生成
    │
    ▼ 【judge ノード】（rag/graph.py）
    │  「回答は十分か？」を Gemini に判定させる
    │  retry_count をインクリメント
    │
    ▼ 【should_retry 分岐】
    ├─ 十分 or retry_count >= 2 ──→ 回答を返す
    └─ 不十分 ──────────────────→ search に戻る（最大2回再試行）
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
パラメータ＋スコアを ./mlruns/ に記録
    │
    ▼ mlflow ui（http://localhost:5000）
実験結果の比較・可視化
```

---

## システム構成

```
study_rag/
├── main.py                  # CLIエントリーポイント
├── app.py                   # Streamlitエントリーポイント
├── pyproject.toml
├── SPEC.md
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py            # PDF読み込み・チャンク分割・DB保存
│   ├── retriever.py         # ChromaDBからの検索
│   ├── chain.py             # LangChainのRAGチェーン
│   ├── graph.py             # LangGraphのフロー定義
│   └── evaluator.py         # 回答品質評価
│
├── mlflow_tracking/
│   └── experiments.py       # MLflow実験ログユーティリティ
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
| `LLM_MODEL` | `gemini-2.5-flash` | 使用するGeminiモデル |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | 埋め込みモデル |

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

## 環境変数

```env
GOOGLE_API_KEY=your_gemini_api_key
MLFLOW_TRACKING_URI=./mlruns
CHROMA_PERSIST_DIR=./chroma_db
```

---

## 非機能要件

- ローカル完結（ChromaDB・MLflowはすべてローカル）
- `.env` ファイルでAPIキーを管理（`.gitignore` 対象）
- `data/pdfs/` と `chroma_db/` は `.gitignore` 対象
- 各フェーズは独立して動作確認できること

---

## 実装順序

1. `rag/ingest.py` — PDF読み込み・ChromaDB保存
2. `rag/retriever.py` + `rag/chain.py` — CLIでの検索・回答
3. `rag/graph.py` — LangGraphフロー
4. `mlflow_tracking/` — 実験ログ・評価
5. `app.py` — Streamlit UI
