"""ChromaDBからの類似検索"""

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

TOP_K = int(os.getenv("TOP_K", "5"))


def retrieve(vectorstore: Chroma, query: str, top_k: int = TOP_K) -> list[Document]:
    """クエリに関連するチャンクをChromaDBから取得する"""
    return vectorstore.similarity_search(query, k=top_k)
