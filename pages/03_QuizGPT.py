import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧠",
)

st.title("QuizGPT")


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f'./.cache/quiz_files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


with st.sidebar:
    choice = st.selectbox("Choose what you want to do", (
        "File",
        "Wikipedia Article",
    ))
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["docx", "txt", "pdf"]
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Enter a topic")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5, lang="ko")
            with st.status("Searching Wikipedia..."):
                doc = retriever.get_relevant_documents(topic)
                st.write(doc)
