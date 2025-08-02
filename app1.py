import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile

st.set_page_config(page_title="PDF Text Extractor", layout="wide")

st.title("📄 PDF Text Extractor & Splitter")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

def load_pdf_file(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        all_docs.extend(documents)
        os.remove(tmp_path)  # Clean up
    return all_docs

def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

if uploaded_files:
    st.info("🔄 Extracting and processing PDF files...")
    extracted_docs = load_pdf_file(uploaded_files)
    split_chunks = text_split(extracted_docs)

    st.success(f"✅ Extracted and split into {len(split_chunks)} chunks.")
    
    # Display a few sample chunks
    for i, chunk in enumerate(split_chunks[:5]):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(chunk.page_content)
