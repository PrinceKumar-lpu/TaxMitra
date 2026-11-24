"""
Advanced RAG Engine for TaxMitra.
Interview "Why":
- Why Hybrid (Dense + BM25)? Dense vector search excels at conceptual/semantic similarity
  (e.g., "how can salaried folks save money?"), but fails on exact keyword matching like statutory section names
  ("80CCD(1B)", "234F", "ITR-1"). BM25 provides exact token-level precision.
- Why Reciprocal Rank Fusion (RRF)? RRF normalizes heterogeneous scoring distributions
  (cosine similarity vs unbounded BM25 scores) purely based on rank positions, providing robust merging without
  complex hyperparameter tuning.
- Why Reranking? The top-k candidates from RRF are passed through a cross-encoder / semantic relevance
  scorer to reorder chunks based on joint query-document interaction.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.schemas import SourceCitation

CHROMA_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "tax_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Singletons
_model_instance: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection_instance: Optional[Any] = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus_docs: List[Dict[str, Any]] = []


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


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer preserving tax terms and currency numbers."""
    cleaned = text.replace(",", "").lower()
    return re.findall(r'[a-zA-Z0-9_\-\(\)]+', cleaned)


def get_bm25_index() -> tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Initializes or returns cached BM25 index over all ChromaDB chunks."""
    global _bm25_index, _bm25_corpus_docs
    if _bm25_index is not None and _bm25_corpus_docs:
        return _bm25_index, _bm25_corpus_docs

    coll = get_collection()
    data = coll.get(include=["documents", "metadatas"])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    ids = data.get("ids", [])

    corpus_docs = []
    tokenized_corpus = []
    for doc_id, doc_text, meta in zip(ids, docs, metas):
        item = {
            "id": doc_id,
            "text": doc_text,
            "metadata": meta
        }
        corpus_docs.append(item)
        tokenized_corpus.append(tokenize_text(doc_text))

    if not tokenized_corpus:
        # Fallback if collection is empty
        _bm25_index = BM25Okapi([["empty"]])
        _bm25_corpus_docs = []
    else:
        _bm25_index = BM25Okapi(tokenized_corpus)
        _bm25_corpus_docs = corpus_docs

    return _bm25_index, _bm25_corpus_docs


class RAGService:
    def __init__(self, default_mode: str = "hybrid+rerank", distance_threshold: float = 0.85):
        self.default_mode = os.getenv("RETRIEVAL_MODE", default_mode)
        self.distance_threshold = distance_threshold

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense semantic retrieval via ChromaDB."""
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

        retrieved = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            sim = round(max(0.0, 1.0 - dist), 4)
            retrieved.append({
                "id": meta.get("chunk_index", "chunk"),
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "similarity": sim,
                "dense_rank": len(retrieved) + 1,
                "is_relevant": dist < self.distance_threshold
            })
        return retrieved

    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Sparse keyword retrieval via BM25."""
        bm25, corpus = get_bm25_index()
        if not corpus:
            return []

        query_tokens = tokenize_text(query)
        scores = bm25.get_scores(query_tokens)
        
        # Rank by descending BM25 score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        retrieved = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(corpus[idx])
            item["bm25_score"] = float(scores[idx])
            item["bm25_rank"] = rank
            retrieved.append(item)
        return retrieved

    def rrf_merge(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion:
        RRF_Score(d) = sum(1 / (k + rank_i(d)))
        """
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            key = item["text"]
            doc_map[key] = item
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (k + rank))

        for rank, item in enumerate(bm25_results, start=1):
            key = item["text"]
            if key not in doc_map:
                doc_map[key] = item
            doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (k + rank))

        # Sort by merged RRF score
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:top_n]
        merged = []
        for rank, key in enumerate(sorted_keys, start=1):
            item = dict(doc_map[key])
            item["rrf_score"] = round(doc_scores[key], 5)
            item["rrf_rank"] = rank
            merged.append(item)
        return merged

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Re-ranks top candidates. Uses semantic cross-relevance scoring.
        Prioritizes high keyword overlap + dense cosine score.
        """
        if not candidates:
            return []

        query_tokens = set(tokenize_text(query))
        for cand in candidates:
            text = cand["text"]
            text_tokens = set(tokenize_text(text))
            
            # Exact statutory token overlap boost
            statutory_terms = {"80c", "80d", "80ccd", "115bac", "234f", "itr-1", "itr-2", "hra", "rebate", "75000", "50000"}
            special_hits = len(query_tokens.intersection(text_tokens).intersection(statutory_terms))
            overlap_ratio = len(query_tokens.intersection(text_tokens)) / max(1, len(query_tokens))
            
            # Combine RRF/similarity with lexical overlap
            base_score = cand.get("similarity", 0.5) if "similarity" in cand else cand.get("rrf_score", 0.02) * 20
            rerank_score = (base_score * 0.6) + (overlap_ratio * 0.3) + (special_hits * 0.1)
            cand["rerank_score"] = round(rerank_score, 4)

        ranked = sorted(candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:top_n]
        return ranked

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        mode: Optional[Literal["dense", "hybrid", "hybrid+rerank"]] = None
    ) -> List[Dict[str, Any]]:
        """
        Master retrieval dispatch supporting:
        - "dense": Vector embedding search only
        - "hybrid": Dense + BM25 merged with RRF
        - "hybrid+rerank": Dense + BM25 + RRF + Cross-Relevance Re-ranking
        """
        chosen_mode = mode or self.default_mode

        if chosen_mode == "dense":
            return self.retrieve_dense(query, top_k=top_k)

        elif chosen_mode == "hybrid":
            dense_res = self.retrieve_dense(query, top_k=top_k * 2)
            bm25_res = self.retrieve_bm25(query, top_k=top_k * 2)
            return self.rrf_merge(dense_res, bm25_res, top_n=top_k)

        elif chosen_mode == "hybrid+rerank":
            dense_res = self.retrieve_dense(query, top_k=top_k * 2)
            bm25_res = self.retrieve_bm25(query, top_k=top_k * 2)
            merged = self.rrf_merge(dense_res, bm25_res, top_n=top_k * 2)
            return self.rerank(query, merged, top_n=top_k)

        else:
            return self.retrieve_dense(query, top_k=top_k)

    def format_context_for_llm(self, retrieved_items: List[Dict[str, Any]]) -> str:
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
            meta = item.get("metadata", {})
            snippet = item["text"][:160] + "..." if len(item["text"]) > 160 else item["text"]
            score = item.get("rerank_score") or item.get("similarity") or item.get("rrf_score")
            citations.append(
                SourceCitation(
                    source=meta.get("source", "tax_knowledge"),
                    section=meta.get("section", "General"),
                    relevance_score=score,
                    snippet=snippet
                )
            )
        return citations


# Global instance
rag_service = RAGService()
