import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os

# 🌐 Set API keys from secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index_name = "pdf-rag"

st.title("📄 Ask Questions About Your PDF")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # 📄 Load and split PDF
    loader = PyPDFLoader("uploaded.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # 🔍 Create Embeddings
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Pinecone.from_documents(docs, embedder, index_name=index_name)

    # 🤖 Load LLM
    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-1b",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    question = st.text_input("Ask a question from your PDF:")
    if question:
        answer = qa.run(question)
        st.write("Answer:", answer)
