# Semantic Source Finder

A Streamlit app for finding relevant source passages in PDF documents using semantic search and reranking.

## Features

- Search across multiple PDF documents
- Retrieve relevant passages with page numbers
- Use multilingual embeddings with BAAI/bge-m3
- Rerank results with BAAI/bge-reranker-v2-m3
- Filter searches by selected documents

## Project structure

```text
app.py              # Streamlit app
ingest.py           # PDF ingestion and vector database creation
requirements.txt    # Python dependencies
documents/          # Local PDF folder, not pushed to GitHub
vector_db/          # Local Chroma database, not pushed to GitHub
