# app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# ====== SETUP ======
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

model = load_model()

# ====== FUNGSI PEMBANTU ======
def transform_data(df):
    df["text"] = df.apply(lambda row:
        f"Produk {row['nama_produk']} dari kategori {row['kategori']} memiliki harga {row['harga']} dan stok {row['stok']} unit.",
        axis=1)
    return df

def build_faiss_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve(query, index, df, top_k=5):
    query_embed = model.encode([query])
    distances, indices = index.search(np.array(query_embed).astype("float32"), top_k)
    return df.iloc[indices[0]]

def generate_answer(query, context, api_key):
    openai.api_key = api_key
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

# ====== STREAMLIT UI ======
# st.set_page_config(page_title="RAG Tabular CSV", page_icon="📦")
st.title("📦 RAG untuk Data Produk CSV")

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV produk", type="csv")

# Input API Key
api_key = st.text_input("Masukkan OpenAI API Key Anda", type="password")

# Query input
query = st.text_input("Masukkan pertanyaanmu:")

# Logika utama
if uploaded_file and query and api_key:
    try:
        df = pd.read_csv(uploaded_file)
        df = transform_data(df)
        index, _ = build_faiss_index(df["text"].tolist())
        
        with st.spinner("🔍 Mencari produk relevan..."):
            results = retrieve(query, index, df)
            context = " ".join(results["text"].tolist())

        st.subheader("📄 Hasil Retrieval:")
        st.dataframe(results[["nama_produk", "kategori", "harga", "stok"]])

        with st.spinner("🧠 Menghasilkan jawaban..."):
            answer = generate_answer(query, context, api_key)

        st.subheader("💬 Jawaban LLM:")
        st.success(answer)
    
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

elif not uploaded_file:
    st.info("📂 Silakan upload file CSV terlebih dahulu.")
elif not api_key:
    st.info("🔑 Masukkan API key Anda.")
