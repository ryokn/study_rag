# システムフロー図

## 全体アーキテクチャ

```mermaid
graph TB
    subgraph Entry["エントリーポイント"]
        CLI["CLI<br/>main.py"]
        WEB["Web UI<br/>app.py / Streamlit"]
        RUNSH["run.sh<br/>インタラクティブメニュー"]
    end

    subgraph Commands["コマンド / モード"]
        INGEST["ingest<br/>PDF取り込み"]
        CHAT["chat<br/>RAGチャット"]
        AGENT["chat --agent<br/>エージェント"]
        TABLE["table<br/>CSV検索"]
        EVAL["eval<br/>品質評価"]
    end

    subgraph Stores["データストア"]
        CHROMA[("ChromaDB<br/>chroma_db/")]
        DUCKDB[("DuckDB<br/>インメモリ")]
        MLFLOW[("MLflow<br/>mlruns.db")]
    end

    subgraph External["外部サービス"]
        LLM["LLM<br/>Gemini / Ollama"]
        DDG["DuckDuckGo<br/>Web検索"]
    end

    RUNSH --> CLI
    CLI --> INGEST
    CLI --> CHAT
    CLI --> AGENT
    CLI --> TABLE
    CLI --> EVAL
    WEB --> CHAT
    WEB --> AGENT
    WEB --> TABLE

    INGEST --> CHROMA
    CHAT --> CHROMA
    AGENT --> CHROMA
    TABLE --> DUCKDB
    EVAL --> CHROMA
    EVAL --> MLFLOW

    CHAT --> LLM
    AGENT --> LLM
    AGENT --> DDG
    TABLE --> LLM
    EVAL --> LLM
```

---

## 1. 取り込みフロー（ingest）

```mermaid
flowchart LR
    PDF["data/pdfs/*.pdf"]
    LOAD["PyPDFLoader<br/>ページ単位で読み込み"]
    SPLIT["RecursiveCharacterTextSplitter<br/>CHUNK_SIZE=500 / OVERLAP=50"]
    EMBED["GoogleGenerativeAIEmbeddings<br/>gemini-embedding-001"]
    CHROMA[("ChromaDB<br/>chroma_db/")]

    PDF --> LOAD --> SPLIT --> EMBED --> CHROMA
```

---

## 2. RAGチャットフロー（chat）

```mermaid
flowchart TD
    Q["ユーザーの質問"]
    HIST["会話履歴<br/>history"]
    SEARCH["search ノード<br/>ChromaDB 類似検索<br/>TOP_K=5件"]
    GEN["generate ノード<br/>コンテキスト＋質問 → LLM → 回答生成"]
    JUDGE["judge ノード<br/>回答品質をLLMが判定<br/>retry_count++"]
    CHECK{"sufficient?\nretry_count >= 2?"}
    RETRY["再検索"]
    ANS["回答を返す"]

    Q --> SEARCH
    HIST --> GEN
    SEARCH --> GEN --> JUDGE --> CHECK
    CHECK -- "十分 or 上限到達" --> ANS
    CHECK -- "不十分 (最大2回)" --> RETRY --> SEARCH
```

---

## 3. エージェントフロー（chat --agent）

```mermaid
flowchart TD
    Q["ユーザーの質問"]
    REACT["ReAct エージェント<br/>rag/agent.py"]

    subgraph Tools["利用可能なツール"]
        T1["search_pdf<br/>ChromaDB でPDF検索"]
        T2["web_search<br/>DuckDuckGo でWeb検索"]
        T3["calculator<br/>数式・数学関数の計算"]
        T4["python_repl<br/>Pythonコード実行"]
    end

    OBS["ツール結果を観察"]
    CHECK{"十分な情報が揃ったか？"}
    ANS["回答を返す"]

    Q --> REACT
    REACT --> Tools
    Tools --> OBS --> CHECK
    CHECK -- "No → 再度ツール選択" --> REACT
    CHECK -- "Yes" --> ANS
```

---

## 4. テーブル検索フロー（table）

```mermaid
flowchart LR
    CSV["data/csv/*.csv"]
    DUCK[("DuckDB<br/>インメモリテーブル")]
    SCHEMA["スキーマ情報取得<br/>テーブル名・カラム定義・行数"]
    Q["ユーザーの自然言語質問"]
    NL2SQL["LLM<br/>NL → SQL 変換"]
    EXEC["DuckDB SQL実行"]
    RESULT["結果 + 実行SQL を返す"]

    CSV --> DUCK
    DUCK --> SCHEMA
    Q --> NL2SQL
    SCHEMA --> NL2SQL
    NL2SQL --> EXEC
    EXEC --> RESULT
```

---

## 5. 評価フロー（eval）

```mermaid
flowchart TD
    JSON["data/eval_questions.json<br/>質問 + 正解リスト"]
    RAG["RAGチェーン実行<br/>質問ごとに回答・コンテキスト取得"]
    RAGAS["RAGAS 評価<br/>faithfulness<br/>answer_relevancy"]
    MLFLOW[("MLflow<br/>mlruns.db")]
    UI["mlflow ui<br/>http://localhost:5000"]

    JSON --> RAG --> RAGAS --> MLFLOW --> UI
```
