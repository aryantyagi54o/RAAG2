import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
from google import genai
import os

st.set_page_config(page_title="🔥 RAG Bot with Gemini", layout="wide")
st.title("RAG Bot ")

# ---------------------------
# 1. Load Embedder (cached)
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------
# 2. Initialize Gemini Client
# ---------------------------

# ❗ IMPORTANT: Put your REAL API KEY here
GEMINI_API_KEY = "AIzaSyBPmgBqZbKUOVFoPrb3BGQSkLMxdZN-WI0"   # ← YOUR KEY HERE

client = genai.Client(api_key="AIzaSyBPmgBqZbKUOVFoPrb3BGQSkLMxdZN-WI0")

# ---------------------------
# 3. PDF Upload + Extraction
# ---------------------------
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf.getbuffer())

    reader = PdfReader(pdf_path)
    text = " ".join([p.extract_text() or "" for p in reader.pages])
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        st.error("PDF has no text!")
        st.stop()

    st.success("PDF loaded!")

    # ---------------------------
    # 4. Chunking
    # ---------------------------
    def chunk_text(text, max_len=350):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, chunk = [], ""
        for s in sentences:
            if len(chunk) + len(s) < max_len:
                chunk += s + " "
            else:
                chunks.append(chunk.strip())
                chunk = s + " "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    chunks = chunk_text(text)

    # ---------------------------
    # 5. Build FAISS Index
    # ---------------------------
    @st.cache_resource
    def build_index(chunks):
        embs = embedder.encode(chunks, show_progress_bar=False).astype("float32")
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        return index, embs

    index, embeddings = build_index(chunks)

    # ---------------------------
    # 6. Multi-turn Chat
    # ---------------------------
    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.text_input("Ask anything from your PDF:")

    if query:
        # -------- Retrieve top chunks --------
        q_vec = embedder.encode([query]).astype("float32")
        D, I = index.search(q_vec, 4)
        context = "\n".join([chunks[i] for i in I[0] if i < len(chunks)])

        # -------- Gemini LLM Call --------
        prompt = f"""
You are an extremely accurate assistant.
Answer ONLY using the text in the context below.
If answer not found, reply exactly: "I don’t know."

Context:
{context}

Question: {query}
Answer:
"""
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite",  # free tier
            contents=prompt
        )
        answer = resp.text

        st.session_state.chat.append(("You", query))
        st.session_state.chat.append(("Bot", answer))

    # ---------------------------
    # 7. Display Chat
    # ---------------------------
    for sender, msg in st.session_state.chat:
        st.markdown(f"**{sender}:** {msg}")
