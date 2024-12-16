import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
    layout="wide",
)


class ChatCallbackHanlder (BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, 'ai')

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHanlder(),
    ]
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=200,
    memory_key="chat_history"
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


def save_message(message, role):
    st.session_state['messages'].append({
        'message': message,
        'role': role,
    })
    # memory.save_context({"input": })


def load_memory(input):
    print(input)
    return memory.load_memory_variables({})['chat_history']


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
        Context: {context}
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


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
    send_message("I'm ready to answer your questions!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask me a question")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(load_memory),
            } | prompt | llm
        )
        with st.chat_message('ai'):
            response = chain.invoke(message)

            st.write(response.content)
            memory.save_context({
                "input": message,
            }, {
                "output": response.content,
            })

    st.write(load_memory({}))


else:
    st.session_state['messages'] = []
