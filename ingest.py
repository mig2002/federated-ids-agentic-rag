"""
Document ingestion script for building the private Pinecone knowledge base from PDF files.
Loads all PDFs from the pdf/ directory, splits chunks, creates the Pinecone index if needed,
and indexes vectors into Pinecone with progress tracking and batching.
"""

import os
import glob
import time
from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    PDF_DIR,
    get_embeddings,
    check_api_keys,
)


def load_pdf_file(pdf_path: str) -> List[Document]:
    """Loads a single PDF file and extracts text page-by-page with metadata."""
    docs = []
    try:
        # Try using pypdf directly for maximum reliability and speed
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "filename": os.path.basename(pdf_path),
                            "page": page_num,
                        },
                    )
                )
    except ImportError:
        # Fallback to langchain PyPDFLoader
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        print(f"  [!] Warning: Error reading {os.path.basename(pdf_path)}: {e}")

    return docs


def load_all_pdfs(directory: str) -> List[Document]:
    """Scans the directory for all .pdf files and loads them."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"PDF directory not found: '{directory}'")

    pdf_files = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: '{directory}'")

    print(f"Found {len(pdf_files)} PDF files in '{directory}'. Loading pages...")

    all_docs = []
    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"[{idx}/{len(pdf_files)}] Reading: {filename[:50]}...")
        doc_pages = load_pdf_file(pdf_path)
        all_docs.extend(doc_pages)
        print(f"       -> Extracted {len(doc_pages)} pages")

    print(f"\nTotal extracted pages across all PDFs: {len(all_docs)}")
    return all_docs


def run_ingestion():
    check_api_keys()

    print("=" * 70)
    print("STEP 1: Loading PDF documents from knowledge base directory")
    print(f"Target folder: {PDF_DIR}")
    print("=" * 70)

    raw_docs = load_all_pdfs(PDF_DIR)
    if not raw_docs:
        print("[!] No documents extracted. Aborting ingestion.")
        return None

    print("\nPreview of first page:")
    print(f"Source: {raw_docs[0].metadata.get('filename')} (Page {raw_docs[0].metadata.get('page')})")
    print(raw_docs[0].page_content[:300] + "...\n")

    print("=" * 70)
    print("STEP 2: Splitting documents into text chunks")
    print("=" * 70)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"Total chunks created: {len(chunks)}")
    if chunks:
        print("\nFirst chunk preview:")
        print(f"Source: {chunks[0].metadata.get('filename')} (Page {chunks[0].metadata.get('page')})")
        print(chunks[0].page_content[:300] + "...\n")

    print("=" * 70)
    print("STEP 3: Preparing Pinecone vector database")
    print("=" * 70)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating serverless index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

        print("Waiting for Pinecone index to be ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' is now ready.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

    print("=" * 70)
    print("STEP 4: Uploading chunks to Pinecone vector store in batches")
    print("=" * 70)

    embeddings = get_embeddings()
    index = pc.Index(PINECONE_INDEX_NAME)
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
    )

    batch_size = 100
    total_chunks = len(chunks)
    print(f"Uploading {total_chunks} chunks in batches of {batch_size}...")

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  -> Uploaded chunks {i + 1} to {min(i + batch_size, total_chunks)} / {total_chunks}")

    print("\n✓ All PDF documents successfully indexed into Pinecone!")
    print(f"  Index: {PINECONE_INDEX_NAME}")
    print(f"  Namespace: {PINECONE_NAMESPACE}")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
