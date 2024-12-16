import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state['messages'] = []


@st.cache_data(show_spinner="Embedding file...")
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


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state['messages'].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], save=False)


st.title("DocumentGPT")

st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an AI about a document.

    Upload a .txt, .pdf, or .docx file at the sidebar to get started.
    """
)

with st.sidebar:
    file = st.file_uploader('Upload a .txt, .pdf, or .docx file', type=[
        'txt', 'pdf', 'docx'])


if file:
    retriever = embed_file(file)
    send_message("I'm ready to answer your questions!", "AI", save=False)
    paint_history()
    message = st.chat_input("Ask me a question")
    if message:
        send_message(message, "Human")
        send_message('I am thinking...', "AI")
else:
    st.session_state['messages'] = []
