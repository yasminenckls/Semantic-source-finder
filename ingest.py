import os
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
DOCUMENTS_FOLDER = "documents"
VECTOR_DB_FOLDER = "vector_db"


def load_pdfs_with_pymupdf(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file_name)
        pdf = fitz.open(file_path)

        print(f"Reading {file_name} with PyMuPDF...")

        for page_index, page in enumerate(pdf):
            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            page_texts = []

            for block_index, block in enumerate(blocks):
                text = block[4].strip()
                if text:
                    text = " ".join(text.split())
                    if len(text) >= 40:
                        page_texts.append(text)

            page_text = "\n\n".join(page_texts)

            if len(page_text.strip()) >= 250:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": file_path,
                            "page": page_index + 1,
                        }
                    )
                )
    return documents


print("Script started")
def main():
    print("Loading PDF documents with PyMuPDF...")
    documents = load_pdfs_with_pymupdf(DOCUMENTS_FOLDER)
    print(f"Loaded {len(documents)} pages from the PDF files.")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, #chaque morceau fait 1400 characteres
    chunk_overlap=250) #avec overlap de 200 au cas ou idées se croisent 

    chunks = text_splitter.split_documents(documents) #je fais ça aux docs 
    print(f"Created {len(chunks)} text chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_FOLDER
    )

    print("Vector database created successfully in the vector_db folder.")

if __name__ == "__main__": #si je lance directement ce fichier avec python ingest.py, alors exécute main()
    main()




