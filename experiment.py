import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re
import os
from google.genai import Client  # ✅ Official Gemini SDK

# ---------------------------
# 0. Streamlit Config
# ---------------------------
st.set_page_config(page_title="🔥 RAG Bot with Gemini 2.5-flash", layout="wide")
st.title("RAG Bot")

# ---------------------------
# 1. Load Embedder (cached)
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------
# 2. Configure Gemini 2.5-flash
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY not found! Add it in Streamlit → Settings → Secrets.")
    st.stop()

client = Client(api_key=GEMINI_API_KEY)

# ---------------------------
# 3. PDF Upload
# ---------------------------
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf.getbuffer())

    reader = PdfReader(pdf_path)
    text = " ".join([page.extract_text() or "" for page in reader.pages])
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        st.error("PDF has no extractable text!")
        st.stop()

    st.success("PDF loaded successfully!")

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
    # 5. Build FAISS Index (cached)
    # ---------------------------
    @st.cache_resource
    def build_index(chunks):
        embeds = embedder.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)
        return index, embeds

    index, embeddings = build_index(chunks)

    # ---------------------------
    # 6. Multi-turn Chat
    # ---------------------------
    if "chat" not in st.session_state:
        st.session_state.chat = []

    query = st.text_input("Ask a question based on your PDF:")

    if query:
        # Retrieve relevant chunks using FAISS
        q_vec = embedder.encode([query]).astype("float32")
        D, I = index.search(q_vec, 4)
        context = "\n".join([chunks[i] for i in I[0]])

        prompt = f"""
You are an extremely accurate assistant.
Answer ONLY using the context below.
If answer is not found, say exactly: "I don’t know."

Context:
{context}

Question: {query}

Answer:
"""

        # ---------------------------
        # 🔥 Gemini 2.5-flash Chat
        # ---------------------------
        chat = client.chats.create(model="gemini-2.5-flash")
        response = chat.send_message(prompt)
        answer = response.text

        st.session_state.chat.append(("You", query))
        st.session_state.chat.append(("Bot", answer))

    # ---------------------------
    # 7. Chat Display
    # ---------------------------
    for sender, msg in st.session_state.chat:
        st.markdown(f"**{sender}:** {msg}")
