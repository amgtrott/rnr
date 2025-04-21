import os
import streamlit as st
import weaviate
import PyPDF2
import requests
from dotenv import load_dotenv
from weaviate.collections.classes.config import Property, DataType, Configure

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Helper: Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Helper: Initialize Weaviate client
def get_weaviate_client():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please check your .env file and reload the app.")

    headers = {
        "X-OpenAI-Api-Key": openai_key,
    }

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url, 
        auth_credentials=weaviate.auth.AuthApiKey(api_key=weaviate_key), 
        headers=headers
    )
    return client

# Helper: Ensure class exists
def ensure_class(client):
    if not client.collections.exists("PDFDocument"):
        client.collections.create(
            name="PDFDocument",
            properties=[
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="content", data_type=DataType.TEXT),
            ],
            vectorizer_config=[
                Configure.NamedVectors.text2vec_weaviate(
                    name="text_vector",
                    source_properties=["content"],
                    model="Snowflake/snowflake-arctic-embed-l-v2.0",
                )
            ],
            generative_config=Configure.Generative.openai(
                model="gpt-3.5-turbo"
            )
        )

# Upload and index PDF
def upload_pdf(file, client):
    text = extract_text_from_pdf(file)
    data_obj = {
        "filename": file.name,
        "content": text,
    }
    client.collections.get("PDFDocument").data.insert(data_obj)
    return "Uploaded and indexed!"

# Generative RAG Search (replaces semantic search)
def generative_search(query, client):
    pdf_collection = client.collections.get("PDFDocument")
    # Use the generative module: RAG with generative-openai (or DeepSeek if supported)
    response = pdf_collection.generate.near_text(
        query=query,
        single_prompt="You are a helpful government assistant. Based on the following rules and regulations, answer the user's question using the content below. If the answer is not found, say so. Content: {content}",
        limit=5
    )
    return response.objects

# Streamlit UI - Modern, Polished Government Website Theme
st.set_page_config(
    page_title="DC Government Rules & Regulations Search",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, elegant, government-style theme with improved layout and colors
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #f4f6fb !important;
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
    }
    .gov-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
    }
    .gov-logo {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        border: 3px solid #003366;
        background: #fff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.2rem;
        font-weight: bold;
        color: #003366;
    }
    .gov-title {
        font-size: 2.2rem;
        font-family: 'Georgia', serif;
        color: #003366;
        margin-bottom: 0;
    }
    .gov-subtitle {
        font-size: 1.1rem;
        color: #005ea2;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    .stTextInput>div>input, .stFileUploader>div>div {
        border: 2px solid #003366 !important;
        background: #fff !important;
    }
    .stButton>button {
        background-color: #005ea2;
        color: #fff;
        border-radius: 4px;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.6rem;
        margin-top: 1rem;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #003366;
    }
    .gov-card {
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 2rem;
        border-left: 6px solid #005ea2;
    }
    .gov-answer {
        background: #eaf3fb;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 1.08rem;
        color: #003366;
        border-left: 4px solid #005ea2;
    }
    .gov-source {
        font-size: 0.97rem;
        color: #555;
        background: #f8f9fa;
        border-radius: 5px;
        padding: 0.7rem;
        margin-top: 0.4rem;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #003366;
        font-family: 'Georgia', serif;
    }
    .stMarkdown a {
        color: #005ea2;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="gov-header"><div class="gov-logo">DC</div><div><h1 class="gov-title">District of Columbia Government</h1></div></div>', unsafe_allow_html=True)
st.markdown('<div class="gov-subtitle">Rules & Regulations Search Portal<br>Easily upload official documents and search for rules, regulations, and legal guidance using advanced AI.</div><hr>', unsafe_allow_html=True)

client = get_weaviate_client()
ensure_class(client)

st.markdown('<div class="gov-card"><h3>📤 Upload Official PDF Document</h3>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF (rules, regulations, etc.)", type="pdf")
if uploaded_file:
    with st.spinner("Processing and indexing document..."):
        msg = upload_pdf(uploaded_file, client)
    st.success(msg)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="gov-card"><h3>🔍 Search Rules & Regulations (AI-Powered)</h3>', unsafe_allow_html=True)
query = st.text_input("Enter your question about DC rules or regulations:", placeholder="e.g. What are the noise regulations for downtown?")
if query:
    with st.spinner("Generating answer from official documents..."):
        results = generative_search(query, client)
    for r in results:
        st.markdown(f'<div class="gov-card">', unsafe_allow_html=True)
        st.markdown(f'<b>📄 {r.properties["filename"]}</b>', unsafe_allow_html=True)
        st.markdown(f'<div class="gov-answer"><b>AI Answer:</b><br>{r.generated}</div>', unsafe_allow_html=True)
        with st.expander("Show source content"):
            st.markdown(f'<div class="gov-source">{r.properties["content"][:1000] + ("..." if len(r.properties["content"]) > 1000 else "")}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
