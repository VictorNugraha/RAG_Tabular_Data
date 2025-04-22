# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# ==== SETUP ====
openai.api_key = "..."  # Ganti dengan API key kamu

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

# ==== LOAD DAN TRANSFORMASI DATA ====
@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df["text"] = df.apply(lambda row:
        f"Produk {row['nama_produk']} dari kategori {row['kategori']} memiliki harga {row['harga']} dan stok {row['stok']} unit.",
        axis=1)
    return df

data = load_data()

@st.cache_resource
def build_faiss_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

index, embeddings = build_faiss_index(data["text"].tolist())

# ==== RETRIEVAL FUNCTION ====
def retrieve(query, top_k=3):
    query_embed = model.encode([query])
    distances, indices = index.search(np.array(query_embed).astype("float32"), top_k)
    return data.iloc[indices[0]]

# ==== LLM FUNCTION ====
def generate_answer(query, context):
    prompt = f"""
    Kamu adalah asisten pintar yang memberikan jawaban berdasarkan informasi produk.

    Pertanyaan: {query}
    Data Produk Terkait:
    {context}

    Jawaban:
    """
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",  
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# ==== STREAMLIT UI ====
st.title("📦 RAG untuk Data Produk Tabular")

query = st.text_input("Masukkan pertanyaanmu:")
if query:
    with st.spinner("🔍 Mencari produk terkait..."):
        results = retrieve(query)
        context = " ".join(results["text"].tolist())

    st.subheader("📄 Hasil Retrieval:")
    st.dataframe(results[["nama_produk", "kategori", "harga", "stok"]])

    with st.spinner("🧠 Menghasilkan jawaban..."):
        answer = generate_answer(query, context)

    st.subheader("💬 Jawaban LLM:")
    st.success(answer)
