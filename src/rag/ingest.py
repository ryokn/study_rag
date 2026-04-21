"""PDFの読み込み・チャンク分割・ChromaDBへの保存"""

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


def load_pdfs(pdf_dir: str) -> list[Document]:
    """指定ディレクトリ内のPDFをすべて読み込む"""
    docs: list[Document] = []
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """ドキュメントをチャンクに分割する"""
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
