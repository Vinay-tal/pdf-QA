import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

# 🚨 Load API keys securely from Streamlit secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 🌐 Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
index_name = "pdf-rag-openai"

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 🎨 Streamlit UI
st.set_page_config(page_title="PDF Q&A with OpenAI", layout="wide")
st.title("📄 Ask Questions from Your PDF")
st.markdown("Upload a PDF and ask any question — powered by OpenAI + Pinecone.")

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # 📑 Load and split PDF
    with st.spinner("Reading and splitting PDF..."):
        loader = PyPDFLoader("uploaded.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

    # 🔍 Generate embeddings
    with st.spinner("Generating embeddings and uploading to Pinecone..."):
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Pinecone.from_documents(docs, embedder, index_name=index_name)

    # 🔮 Load OpenAI LLM
    llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

    # 🧠 Build QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # ❓ Ask a question
    st.subheader("Ask a question about the PDF:")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.run(query)
            st.success("💡 Answer:")
            st.write(result)
