"""Streamlit Web UIエントリーポイント"""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="study-rag", layout="wide")
st.title("RAG チャット")


@st.cache_resource
def get_vectorstore():
    from rag.ingest import load_vectorstore
    return load_vectorstore()


def ingest_uploaded_pdf(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    from langchain_community.document_loaders import PyPDFLoader
    from rag.ingest import split_documents, build_vectorstore

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        chunks = split_documents(docs)
        build_vectorstore(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# サイドバー: 設定・PDFアップロード
with st.sidebar:
    st.header("モード設定")
    agent_mode = st.toggle("エージェントモード", value=False, help="PDF検索・Web検索・計算・コード実行ツールを自律的に使用します")

    st.header("PDFを取り込む")
    uploaded = st.file_uploader("PDFファイルを選択", type="pdf")
    if uploaded and st.button("取り込む"):
        with st.spinner("取り込み中..."):
            ingest_uploaded_pdf(uploaded)
            st.cache_resource.clear()
        st.success("取り込み完了")

# チャット
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("検索・回答中..."):
            vectorstore = get_vectorstore()
            history = [
                (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
                for i in range(0, len(st.session_state.messages) - 1, 2)
                if st.session_state.messages[i]["role"] == "user"
                and st.session_state.messages[i + 1]["role"] == "assistant"
            ]
            if agent_mode:
                from rag.agent import run_agent
                answer = run_agent(vectorstore, question, history)
            else:
                from rag.graph import run_graph
                answer = run_graph(vectorstore, question, history)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
