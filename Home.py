import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖",
    layout="wide",
)

st.title('Home')

with st.sidebar:
    st.title('Sidebar')

tab_one, tab_two = st.tabs(["A", "B"])

with tab_one:
    st.write("This is tab A")

with tab_two:
    st.write("This is tab B")
