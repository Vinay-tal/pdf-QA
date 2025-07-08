import streamlit as st
import os
import openai
import fitz  # PyMuPDF
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# 🔐 Load API keys from Streamlit secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# 📦 Pinecone setup (SDK v3)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pdf-rag"
region = "us-east-1"

# ⏳ Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region)
    )

index = pc.Index(index_name)
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


# 🔍 Manual retrieval
def retrieve_context(query):
    query_vector = embedder.embed_query(query)
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    return " ".join([m['metadata']['text'] for m in results['matches']])


# 🧠 Generate answer with OpenAI
def ask_openai(question, context):
    prompt = f"""Use the following context to answer the question:
{context}

Question: {question}
Answer:"""
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    ).choices[0].message.content.strip()


# 🧩 Text splitter
def split_text(text, max_len=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        chunks.append(text[start:end])
        start += max_len - overlap
    return chunks


# 🎨 Streamlit UI
st.set_page_config(page_title="📄 PDF Q&A Manual", layout="wide")
st.title("📄 Ask Questions About Your PDF (Manual Pinecone + OpenAI)")
st.markdown("Upload a PDF and ask any question. Fully custom using Pinecone v3 and OpenAI.")

uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("📚 Reading and embedding PDF..."):
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()

        chunks = split_text(text)

        for chunk in chunks:
            vector = embedder.embed_query(chunk)
            uid = str(uuid4())
            index.upsert([
                {
                    "id": uid,
                    "values": vector,
                    "metadata": {"text": chunk}
                }
            ])
    st.success("✅ PDF processed and indexed!")

    # ❓ Ask questions
    st.subheader("Ask a question:")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("🤖 Generating answer..."):
            context = retrieve_context(query)
            answer = ask_openai(query, context)
            st.success("💡 Answer:")
            st.write(answer)
