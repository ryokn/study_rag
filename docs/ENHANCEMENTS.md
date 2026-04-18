# エンハンス計画メモ

学習目的で実装したい機能アイデア。

---

## 1. DuckDB + Delta Table を使った構造化データ検索

- 非構造化テキスト（PDF）だけでなく、CSV・Parquet等の構造化データも検索対象にする
- DuckDB でインメモリSQL検索、Delta Table でデータのバージョン管理
- LangChain の SQLDatabaseChain や Tool と組み合わせてRAGから自然言語でクエリ実行

**学べること**
- DuckDB / Delta Lake の基本操作
- テキスト検索と構造化データ検索のハイブリッドRAG

---

## 2. Agentic AI の実装

- 単純な検索→回答だけでなく、複数ツールを自律的に選択・実行するエージェントを実装
- LangGraph のマルチエージェント構成、またはReActパターンで実装
- ツール例: PDF検索・Web検索・計算・コード実行

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
