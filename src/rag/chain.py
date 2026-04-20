"""LangChainを使ったRAGチェーンの定義

LangChain Expression Language (LCEL) によるチェーンの組み立て方:
  retriever → _format_docs → プロンプト → LLM → StrOutputParser

パイプ演算子 `|` で各コンポーネントを繋ぐことで、
入力が左から右へ順番に流れる処理パイプラインを構築できる。
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.llm import build_llm

# プロンプトテンプレート: {context}と{question}が実行時に埋め込まれる
# 「コンテキストのみ」と明示することでLLMの幻覚（ハルシネーション）を抑制する
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """以下のコンテキストのみを使って質問に答えてください。
コンテキストに答えが含まれない場合は「わかりません」と答えてください。

コンテキスト:
{context}

質問: {question}

回答:"""
)


def _format_docs(docs: list[Document]) -> str:
    """Documentリストを改行区切りの文字列に変換する。

    LLMに渡すコンテキストとして使用するため、
    各チャンクのテキスト（page_content）を結合する。
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: Chroma) -> Runnable:
    """RAGチェーンを構築して返す。

    チェーンの流れ:
      1. retrieverが質問に関連するチャンクをChromaDBから取得
      2. _format_docsでチャンクを文字列に結合（context）
      3. RunnablePassthroughで質問文をそのまま通す（question）
      4. PROMPT_TEMPLATEにcontext・questionを埋め込む
      5. LLMで回答を生成
      6. StrOutputParserでAIMessageから文字列を取り出す
    """
    llm = build_llm()
    # as_retriever()でChromaをLangChainのRetrieverインターフェースに変換
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain
