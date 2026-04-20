"""DuckDBを使ったCSVファイルへの自然言語クエリ

処理フロー:
  1. load_csv_tables(): data/csv/*.csv を DuckDB のインメモリテーブルに読み込む
                        ← 起動時1回のみ実行（その後はメモリ上で高速クエリ）
  2. get_schema_info(): テーブル名・カラム定義・行数を文字列化してLLMに渡す
  3. nl_to_sql()      : LLMが自然言語の質問をDuckDB用SQLに変換する
  4. query_tables()   : SQLを実行してDataFrameを文字列で返す

DuckDB の特徴:
  - インメモリSQLエンジン（サーバー不要、Pythonライブラリとして動く）
  - CSV/Parquetファイルをそのままテーブルとして扱える（read_csv_auto）
  - pandasとの相互変換が容易（fetchdf()）
"""

import os
import re
from pathlib import Path

import duckdb

# CSVディレクトリのデフォルトパス（環境変数で上書き可能）
CSV_DIR = os.getenv("CSV_DIR", "./data/csv")


def load_csv_tables(csv_dir: str = CSV_DIR) -> duckdb.DuckDBPyConnection:
    """CSVディレクトリ内の全ファイルをDuckDBのインメモリテーブルに読み込む。

    read_csv_auto() は列の型を自動推論するため、スキーマ定義が不要。
    テーブル名はCSVファイルのステム（拡張子なしのファイル名）になる。
    例: japan_prefectures.csv → テーブル名 japan_prefectures

    Args:
        csv_dir: CSVファイルが置かれているディレクトリのパス

    Returns:
        テーブルが登録済みのDuckDB接続オブジェクト
    """
    # duckdb.connect() 引数なしでインメモリDBを作成（ファイルには保存しない）
    conn = duckdb.connect()
    csv_path = Path(csv_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSVディレクトリが見つかりません: {csv_dir}")

    loaded: list[str] = []
    for csv_file in sorted(csv_path.glob("*.csv")):
        table_name = csv_file.stem
        # CSVを SELECT * でそのままテーブルとしてインポート
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
        loaded.append(table_name)

    if not loaded:
        raise ValueError(f"CSVファイルが見つかりません: {csv_dir}")

    return conn


def get_schema_info(conn: duckdb.DuckDBPyConnection) -> str:
    """全テーブルのスキーマ情報を人間が読みやすい文字列で返す。

    LLMへのプロンプトに埋め込むことで、LLMが正しいSQL（テーブル名・列名）を
    生成できるようにするために使用する。
    """
    tables = conn.execute("SHOW TABLES").fetchall()
    lines: list[str] = []
    for (table_name,) in tables:
        columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
        # DESCRIBE は [(列名, 型, Null可否, ...), ...] を返す
        col_defs = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        lines.append(f"テーブル: {table_name}  ({row_count}行)\n  カラム: {col_defs}")
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    """LLMのレスポンステキストからSQLクエリ部分のみを抽出する。

    LLMは説明文やMarkdownのコードブロックと一緒にSQLを返すことがあるため、
    以下の優先順位で抽出を試みる:
      1. ```sql ... ``` または ``` ... ``` のMarkdownコードブロック
      2. SELECT / WITH で始まる行
      3. どちらでもなければそのままの文字列を返す
    """
    # Markdownのコードブロックを優先して抽出
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # コードブロックがない場合はSELECT/WITH始まりの行を探す
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SELECT") or stripped.upper().startswith("WITH"):
            return stripped

    return text.strip()


def nl_to_sql(question: str, schema_info: str) -> str:
    """自然言語の質問をDuckDB用SQLクエリに変換する。

    LLMにスキーマ情報（テーブル名・列名・型）を与えることで、
    正しい列名・テーブル名を使ったSQLを生成させる。

    Args:
        question: ユーザーの自然言語による質問
        schema_info: get_schema_info()が返すテーブル定義文字列

    Returns:
        実行可能なDuckDB SQLクエリ文字列
    """
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

    # AnthropicモデルはcontentをTextBlockリストで返す場合がある
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))

    return _extract_sql(content)


def query_tables(
    conn: duckdb.DuckDBPyConnection,
    question: str,
    max_rows: int = 20,
) -> str:
    """自然言語の質問をSQLに変換してDuckDBで実行し、結果を文字列で返す。

    Args:
        conn: load_csv_tables()で作成したDuckDB接続オブジェクト
        question: ユーザーの自然言語による質問
        max_rows: 表示する最大行数（大量結果の際の上限）

    Returns:
        クエリ結果の表形式文字列 + 実行したSQLクエリ
    """
    schema_info = get_schema_info(conn)
    sql = nl_to_sql(question, schema_info)

    try:
        # fetchdf() でpandasのDataFrameとして結果を取得
        df = conn.execute(sql).fetchdf()
        if df.empty:
            return f"結果: 0件\n実行SQL: {sql}"

        result_str = df.head(max_rows).to_string(index=False)
        # 全件数がmax_rowsを超える場合は件数情報を付加
        suffix = (
            f"\n... ({len(df)}件中{min(max_rows, len(df))}件表示)"
            if len(df) > max_rows
            else f"\n({len(df)}件)"
        )
        return f"{result_str}{suffix}\n\n実行SQL: {sql}"
    except Exception as e:
        # SQL実行エラーの場合は生成されたSQLも一緒に返してデバッグに役立てる
        return f"クエリエラー: {e}\n生成されたSQL: {sql}"
