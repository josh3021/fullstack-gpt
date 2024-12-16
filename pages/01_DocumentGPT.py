import stat
import time

import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f'./.cache/embeddings/{file.name}')
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader('./files/chapter_one.txt')
    docs = loader.load_and_split(text_splitter=splitter)
    embeddiings = OpenAIEmbeddings()
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddiings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about a document.
    """
)

file = st.file_uploader('Upload a .txt, .pdf, or .docx file', type=[
                        'txt', 'pdf', 'docx'])


if file:
    retriever = embed_file(file)
    s = retriever.invoke('winston')
    st.write(s)
