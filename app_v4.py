# app_v2_general_sidebar_button.py

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
    def row_to_text(row):
        parts = [f"{col}: {row[col]}" for col in df.columns]
        return " | ".join(parts)
    df["text"] = df.apply(row_to_text, axis=1)
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
    Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data berikut.

    Pertanyaan: {query}

    Data yang relevan:
    {context}

    Jawaban:
    """
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# ====== STREAMLIT UI ======
# st.set_page_config(page_title="RAG CSV Umum", layout="wide")
st.title("📊 RAG CSV Umum (Tanpa Struktur Khusus)")

# ====== SIDEBAR ======
st.sidebar.header("🔧 Pengaturan")

uploaded_file = st.sidebar.file_uploader("📂 Upload file CSV", type="csv")
input_api_key = st.sidebar.text_input("🔑 Masukkan OpenAI API Key", type="password")
activate_api = st.sidebar.button("🔒 Aktifkan API Key")

# Simpan state aktivasi
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if activate_api and input_api_key:
    st.session_state.api_key = input_api_key
    st.sidebar.success("✅ API Key aktif!")

# ====== MAIN INPUT ======
query = st.text_input("❓ Masukkan pertanyaan Anda")
run_query = st.button("🚀 Jawab Pertanyaan")

# ====== PROSES ======
if uploaded_file and run_query and st.session_state.api_key:
    try:
        df = pd.read_csv(uploaded_file)
        df = transform_data(df)
        index, _ = build_faiss_index(df["text"].tolist())

        with st.spinner("🔍 Mencari data relevan..."):
            results = retrieve(query, index, df)
            context = "\n".join(results["text"].tolist())

        st.subheader("📄 Data yang Ditemukan:")
        st.dataframe(results.drop(columns=["text"]))

        with st.spinner("🧠 Menghasilkan jawaban..."):
            answer = generate_answer(query, context, st.session_state.api_key)

        st.subheader("💬 Jawaban:")
        st.success(answer)

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

elif run_query and not uploaded_file:
    st.warning("📂 Silakan upload file CSV terlebih dahulu.")
elif run_query and not st.session_state.api_key:
    st.warning("🔐 Anda harus mengaktifkan API Key terlebih dahulu.")
