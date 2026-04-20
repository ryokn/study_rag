"""DuckDBを使ったCSVファイルへの自然言語クエリ"""

import os
import re
from pathlib import Path

import duckdb

CSV_DIR = os.getenv("CSV_DIR", "./data/csv")


def load_csv_tables(csv_dir: str = CSV_DIR) -> duckdb.DuckDBPyConnection:
    """CSVディレクトリ内のファイルをDuckDBに読み込む"""
    conn = duckdb.connect()
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSVディレクトリが見つかりません: {csv_dir}")

    loaded = []
    for csv_file in sorted(csv_path.glob("*.csv")):
        table_name = csv_file.stem
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
        loaded.append(table_name)

    if not loaded:
        raise ValueError(f"CSVファイルが見つかりません: {csv_dir}")

    return conn


def get_schema_info(conn: duckdb.DuckDBPyConnection) -> str:
    """全テーブルのスキーマ情報を文字列で返す"""
    tables = conn.execute("SHOW TABLES").fetchall()
    lines = []
    for (table_name,) in tables:
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        col_defs = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        lines.append(f"テーブル: {table_name}  ({row_count}行)\n  カラム: {col_defs}")
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    """LLMレスポンスからSQLを抽出する"""
    # ```sql ... ``` または ``` ... ``` ブロックを抽出
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # セミコロンで終わる行を探す
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SELECT") or stripped.upper().startswith("WITH"):
            return stripped
    return text.strip()


def nl_to_sql(question: str, schema_info: str) -> str:
    """自然言語の質問をDuckDB用SQLに変換する"""
    from rag.llm import build_base_llm

    llm = build_base_llm()
    prompt = f"""以下のDuckDBテーブル定義を参考に、質問に対するSQLクエリを生成してください。
SQLクエリのみを返してください（説明や前置きは不要です）。

=== テーブル定義 ===
{schema_info}

=== 質問 ===
{question}

=== SQL ==="""

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return _extract_sql(content)


def query_tables(
    conn: duckdb.DuckDBPyConnection,
    question: str,
    max_rows: int = 20,
) -> str:
    """自然言語でCSVテーブルを検索して結果を返す"""
    schema_info = get_schema_info(conn)
    sql = nl_to_sql(question, schema_info)

    try:
        df = conn.execute(sql).fetchdf()
        if df.empty:
            return f"結果: 0件\n実行SQL: {sql}"
        result_str = df.head(max_rows).to_string(index=False)
        suffix = f"\n... ({len(df)}件中{min(max_rows, len(df))}件表示)" if len(df) > max_rows else f"\n({len(df)}件)"
        return f"{result_str}{suffix}\n\n実行SQL: {sql}"
    except Exception as e:
        return f"クエリエラー: {e}\n生成されたSQL: {sql}"
