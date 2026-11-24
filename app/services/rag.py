"""
RAG Service for TaxMitra.
Connects to local ChromaDB vector store.
Embeds user queries with all-MiniLM-L6-v2 and retrieves top-k chunks.
Enforces strict hallucination-control:
- If similarity distance exceeds threshold, returns low-confidence indicator.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.schemas import SourceCitation

CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "tax_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazy-loaded singletons to optimize startup performance
_model_instance: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection_instance: Optional[Any] = None


def get_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def get_collection():
    global _chroma_client, _collection_instance
    if _collection_instance is None:
        if not CHROMA_PERSIST_DIR.exists():
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        _collection_instance = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection_instance


class RAGService:
    def __init__(self, distance_threshold: float = 0.85):
        """
        distance_threshold: In cosine distance (0.0 means identical, 1.0 means orthogonal, >1 means negative).
        Any distance >= 0.85 indicates negligible semantic relevance.
        """
        self.distance_threshold = distance_threshold

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k context chunks from ChromaDB for the given user query.
        Returns list of dicts with text, metadata, and relevance_score.
        """
        collection = get_collection()
        if collection.count() == 0:
            return []

        model = get_embedding_model()
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Cosine similarity roughly (1 - distance) when normalized
            similarity = round(max(0.0, 1.0 - dist), 4)
            retrieved_items.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": similarity,
                "is_relevant": dist < self.distance_threshold
            })

        return retrieved_items

    def format_context_for_llm(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """Builds structured context markdown block for prompt."""
        if not retrieved_items:
            return "No matching tax documents found in knowledge base."

        context_blocks = []
        for i, item in enumerate(retrieved_items, start=1):
            src = item["metadata"].get("source", "unknown")
            sec = item["metadata"].get("section", "General")
            text = item["text"]
            context_blocks.append(f"--- Document [{i}]: {src} | Section: {sec} ---\n{text}")

        return "\n\n".join(context_blocks)

    def to_source_citations(self, retrieved_items: List[Dict[str, Any]]) -> List[SourceCitation]:
        citations = []
        for item in retrieved_items:
            # Only cite if sufficiently relevant
            if item.get("is_relevant", True):
                meta = item["metadata"]
                snippet = item["text"][:160] + "..." if len(item["text"]) > 160 else item["text"]
                citations.append(
                    SourceCitation(
                        source=meta.get("source", "knowledge_base"),
                        section=meta.get("section", "General"),
                        relevance_score=item.get("similarity"),
                        snippet=snippet
                    )
                )
        return citations


# Global RAG service instance
rag_service = RAGService()
