import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ✅ Load API keys from secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "pdf-rag-openai"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait for index to be ready
    import time
    while True:
        status = pc.describe_index(index_name).status['ready']
        if status:
            break
        time.sleep(1)

# Now safely access it
index = pc.Index(index_name)

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

    with st.spinner("📚 Reading and splitting PDF..."):
        loader = PyPDFLoader("uploaded.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)

    with st.spinner("🔗 Creating vectorstore with OpenAI Embeddings..."):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        index = pc.Index(index_name)
        
        vectorstore = LangchainPinecone.from_documents(
            docs,
            embedding=embeddings,
            index=index,
            text_key="text"
        )

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
