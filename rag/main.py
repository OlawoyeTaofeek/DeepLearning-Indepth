import streamlit as st
from openai import OpenAI
import numpy as np
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🔹 PDF LOADER
# -----------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# -----------------------------
# 🔹 BETTER CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


# -----------------------------
# 🔹 EMBEDDINGS
# -----------------------------
def embed_chunks(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return [item.embedding for item in response.data]


# -----------------------------
# 🔹 SIMILARITY
# -----------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# 🔹 RETRIEVAL
# -----------------------------
def retrieve(query, chunks, embeddings, top_k=3):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((chunks[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scores[:top_k]]


# -----------------------------
# 🔹 STREAMING RESPONSE (WITH MEMORY)
# -----------------------------
def stream_answer(messages, context):
    system_prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer.

If the answer is not in the context, say "I don't know".

Context:
{context}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            *messages
        ],
        stream=True
    )

    full_text = ""
    placeholder = st.empty()

    for event in response:
        if event.type == "response.output_text.delta":
            full_text += event.delta
            placeholder.markdown(full_text)

    return full_text


# -----------------------------
# 🔹 STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("📄 Chat with your PDF (RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# -----------------------------
# 🔹 SESSION STATE
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.embeddings = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 🔹 CLEAR CHAT BUTTON
# -----------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []

# -----------------------------
# 🔹 PROCESS PDF
# -----------------------------
if uploaded_file is not None and st.session_state.chunks is None:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

    st.success("✅ PDF processed!")

# -----------------------------
# 🔹 DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 🔹 CHAT INPUT
# -----------------------------
if st.session_state.chunks:
    query = st.chat_input("Ask a question about the PDF")

    if query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            retrieved_chunks = retrieve(
                query,
                st.session_state.chunks,
                st.session_state.embeddings
            )

            context = "\n\n".join(retrieved_chunks)

            # Limit memory (VERY IMPORTANT)
            MAX_HISTORY = 10
            messages = st.session_state.messages[-MAX_HISTORY:]

            # Generate response
            with st.chat_message("assistant"):
                answer = stream_answer(messages, context)

            # Save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            # Show sources
            with st.expander("📚 Sources"):
                for chunk in retrieved_chunks:
                    st.write(chunk[:300] + "...")