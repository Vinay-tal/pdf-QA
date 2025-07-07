import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

# ✅ Load API keys from secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")  # Use your Pinecone env
index_name = "pdf-rag-openai"

# 🎨 Streamlit UI
st.set_page_config(page_title="📄 PDF Q&A with OpenAI", layout="wide")
st.title("📄 Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask any question. Powered entirely by OpenAI + Pinecone.")

# 📄 Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded!")

    # 📑 Load and split
    with st.spinner("🔍 Reading and chunking PDF..."):
        loader = PyPDFLoader("uploaded.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

    # 🧠 OpenAI Embeddings
    with st.spinner("🔗 Creating vectorstore with OpenAI Embeddings..."):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    # 🔮 OpenAI LLM
    llm = OpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

    # ⚙️ QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # ❓ Ask Question
    st.subheader("Ask a question:")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("🤖 Thinking..."):
            answer = qa_chain.run(query)
            st.success("💡 Answer:")
            st.write(answer)
