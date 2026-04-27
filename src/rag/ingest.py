"""PDFの読み込み・チャンク分割・ChromaDBへの保存

PDFローダーの選択:
  環境変数 PDF_LOADER で切り替え可能。

  pypdf（デフォルト）:
    LangChain 標準の PyPDFLoader を使用。
    シンプルなテキスト抽出に適しているが、
    表がスペース区切りの崩れたテキストになる問題がある。

  pymupdf4llm:
    MuPDF ベースの高精度パーサー。PDF を Markdown 形式に変換する。
    表を |col|col| 形式で保持し、段組レイアウトも正しく処理できる。
    技術文書・仕様書・研究論文など表や図を含むPDFに適している。
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.llm import build_embeddings

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "study_rag"

# 無料枠のレート制限: 100リクエスト/分
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "80"))
EMBED_BATCH_INTERVAL = int(os.getenv("EMBED_BATCH_INTERVAL", "65"))

# PDFローダーの選択: pypdf（デフォルト）または pymupdf4llm
PDF_LOADER = os.getenv("PDF_LOADER", "pypdf")


def _load_pdf_pypdf(pdf_path: Path) -> list[Document]:
    """PyPDFLoader でPDFをページ単位のDocumentリストとして読み込む。

    シンプルなテキスト抽出。表構造は保持されない。
    """
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def _load_pdf_pymupdf4llm(pdf_path: Path) -> list[Document]:
    """pymupdf4llm でPDFをMarkdown形式のDocumentリストとして読み込む。

    page_chunks=True にすることでページごとの Document が得られる。
    各ページの dict は以下の構造:
      {
        "text": "# 見出し\n| 列1 | 列2 |\n...",  # Markdown テキスト
        "metadata": {"file_path": ..., "page": ..., ...}
      }

    表は |col|col| 形式で保持されるため、テキスト抽出より高精度。
    """
    import pymupdf4llm

    # page_chunks=True でページ単位の dict リストを返す
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return [
        Document(
            page_content=page["text"],
            metadata={**page["metadata"], "source": str(pdf_path)},
        )
        for page in pages
        # 空ページ（画像のみ等）はスキップ
        if page["text"].strip()
    ]


def load_pdfs(pdf_dir: str) -> list[Document]:
    """指定ディレクトリ内のPDFをすべて読み込む。

    PDF_LOADER 環境変数に従ってローダーを切り替える。
    """
    docs: list[Document] = []
    loader_name = PDF_LOADER

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        print(f"  ローダー: {loader_name} / ファイル: {pdf_path.name}")
        if loader_name == "pymupdf4llm":
            docs.extend(_load_pdf_pymupdf4llm(pdf_path))
        else:
            docs.extend(_load_pdf_pypdf(pdf_path))

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """ドキュメントをチャンクに分割する。

    pymupdf4llm が返す Markdown テキストは表・見出しの区切りが明確なため、
    RecursiveCharacterTextSplitter の区切り文字（改行・空行）が有効に機能する。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """チャンクをEmbeddingしてChromaDBに保存する。

    無料枠のレート制限（100リクエスト/分）に対応するため、
    EMBED_BATCH_SIZE 件ごとに分割して EMBED_BATCH_INTERVAL 秒待機しながら保存する。
    Embeddingプロバイダーは LLM_PROVIDER 環境変数に従い build_embeddings() が選択する。
    """
    embeddings = build_embeddings()
    vectorstore: Chroma | None = None
    total = len(chunks)

    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        print(f"  バッチ {batch_num}/{total_batches} ({len(batch)} チャンク) を保存中...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

        if i + EMBED_BATCH_SIZE < total:
            print(f"  レート制限対応のため {EMBED_BATCH_INTERVAL} 秒待機中...")
            time.sleep(EMBED_BATCH_INTERVAL)

    assert vectorstore is not None
    return vectorstore


def ingest(pdf_dir: str = "./data/pdfs") -> Chroma:
    """PDFの取り込みからChromaDB保存までを一括実行する"""
    print(f"PDFを読み込み中: {pdf_dir}")
    docs = load_pdfs(pdf_dir)
    if not docs:
        raise ValueError(f"PDFが見つかりません: {pdf_dir}")
    print(f"  {len(docs)} ページ読み込み完了")

    chunks = split_documents(docs)
    print(f"  {len(chunks)} チャンクに分割完了")

    print("ChromaDBに保存中...")
    vectorstore = build_vectorstore(chunks)
    print("  保存完了")

    return vectorstore


def load_vectorstore() -> Chroma:
    """既存のChromaDBを読み込む"""
    embeddings = build_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
