"""
Ingestion script for TaxMitra RAG knowledge base.
- Recursively scans data/docs/*.md
- Chunks markdown with header awareness and sliding overlap
- Generates dense vector embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Stores vectors and rich metadata (file source, section title, doc_id) in ChromaDB
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "tax_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def split_markdown_into_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits markdown by headings first, then applies sliding-window character/word chunking.
    Ensures that headers and section context are preserved in each chunk metadata.
    """
    sections = re.split(r'\n(?=#{1,3}\s+)', text)
    chunks = []
    
    for section_idx, section in enumerate(sections):
        section_lines = section.strip().split("\n")
        header = "General"
        if section_lines and section_lines[0].startswith("#"):
            header = section_lines[0].lstrip("#").strip()

        words = section.strip().split()
        if not words:
            continue

        # If section is small enough, keep as single chunk
        if len(words) <= chunk_size:
            chunk_text = " ".join(words)
            chunk_id = f"{source_filename}_{section_idx}_0"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "section": header,
                    "chunk_index": len(chunks)
                }
            })
        else:
            start = 0
            sub_idx = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_slice = words[start:end]
                # Prepend section context for better semantic embedding
                chunk_text = f"[{header}] " + " ".join(chunk_slice)
                chunk_id = f"{source_filename}_{section_idx}_{sub_idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source_filename,
                        "section": header,
                        "chunk_index": len(chunks)
                    }
                })
                sub_idx += 1
                if end == len(words):
                    break
                start += (chunk_size - chunk_overlap)

    return chunks


def ingest_tax_docs(reset_collection: bool = True):
    print(f"=== Starting TaxMitra Ingestion ===")
    print(f"Docs Directory: {DOCS_DIR}")
    print(f"Chroma DB Path: {CHROMA_PERSIST_DIR}")
    
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs folder not found: {DOCS_DIR}")

    # Initialize Chroma client
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    if reset_collection:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"Cleared existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    doc_files = list(DOCS_DIR.glob("*.md"))
    print(f"Found {len(doc_files)} markdown documents to ingest.")

    all_chunks = []
    for doc_path in doc_files:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        file_chunks = split_markdown_into_chunks(content, doc_path.name)
        all_chunks.extend(file_chunks)
        print(f"  - {doc_path.name}: {len(file_chunks)} chunks")

    if not all_chunks:
        print("No chunks generated. Exiting.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("Inserting chunks into ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Ingestion successful! {collection.count()} chunks stored in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest tax documents into ChromaDB")
    parser.add_argument("--no-reset", action="store_true", help="Do not drop existing collection")
    args = parser.parse_args()
    ingest_tax_docs(reset_collection=not args.no_reset)
