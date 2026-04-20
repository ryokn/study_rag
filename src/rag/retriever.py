"""ChromaDBからの類似検索

ベクトル類似検索（Similarity Search）の仕組み:
  1. クエリテキストをEmbedding（数値ベクトル）に変換
  2. ChromaDB内の全チャンクとコサイン類似度を計算
  3. スコアが高い上位TOP_K件を返す

これにより「意味的に近い」チャンクを取得できる
（キーワード完全一致ではなく意味の近さで検索）。
"""

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

# 検索で取得するチャンク数。多いほど文脈が豊富だがLLMへの入力が増える
TOP_K = int(os.getenv("TOP_K", "5"))


def retrieve(vectorstore: Chroma, query: str, top_k: int = TOP_K) -> list[Document]:
    """クエリに意味的に近いチャンクをChromaDBから取得して返す。

    Args:
        vectorstore: 取り込み済みのChromaDBインスタンス
        query: 検索クエリ（ユーザーの質問文）
        top_k: 取得する上位件数（デフォルトはTOP_K環境変数）

    Returns:
        関連チャンクのDocumentリスト（page_content + metadataを持つ）
    """
    return vectorstore.similarity_search(query, k=top_k)
