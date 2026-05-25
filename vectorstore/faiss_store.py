"""FAISS Vector Store Implementation"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
from loguru import logger

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu")


class FAISSVectorStore:
    """FAISS-based vector store for document embeddings"""

    def __init__(self, dimension: int = 512, index_type: str = "flat"):
        """
        Initialize FAISS vector store

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index(index_type)
        self.documents = []  # Store document texts
        self.metadata = []   # Store document metadata
        self.id_map = {}     # Map internal IDs to document IDs

        logger.info(f"Initialized FAISS vector store with dimension={dimension}, type={index_type}")

    def _create_index(self, index_type: str) -> faiss.Index:
        """Create FAISS index based on type"""
        if index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity after normalization)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
        elif index_type == "hnsw":
            return faiss.IndexHNSWFlat(self.dimension, 32)  # M=32
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        metadata: Optional[List[dict]] = None
    ):
        """
        Add documents with embeddings to the store

        Args:
            embeddings: Document embeddings (n x dimension)
            documents: List of document texts
            metadata: Optional list of metadata dicts
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        # Normalize embeddings
        embeddings = self._normalize_vectors(embeddings.astype(np.float32))

        # Add to FAISS index
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        # Store documents and metadata
        start_id = len(self.documents)
        self.documents.extend(documents)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])

        # Update ID mapping
        for i in range(len(documents)):
            doc_id = f"doc_{start_id + i}"
            self.id_map[start_id + i] = doc_id

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, dict, float]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (document, metadata, score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Normalize query
        query_embedding = self._normalize_vectors(
            query_embedding.reshape(1, -1).astype(np.float32)
        )

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            if score < threshold:
                continue

            doc = self.documents[idx]
            meta = self.metadata[idx].copy()
            meta["doc_id"] = self.id_map.get(idx, f"doc_{idx}")

            results.append((doc, meta, float(score)))

        logger.info(f"Found {len(results)} results for query")
        return results

    def save(self, directory: str):
        """Save vector store to directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(directory / "faiss.index"))

        # Save documents and metadata
        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        with open(directory / "id_map.json", "w") as f:
            json.dump(self.id_map, f)

        # Save config
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_documents": len(self.documents)
        }
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Saved vector store to {directory}")

    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        """Load vector store from directory"""
        directory = Path(directory)

        # Load config
        with open(directory / "config.json", "r") as f:
            config = json.load(f)

        # Create store
        store = cls(
            dimension=config["dimension"],
            index_type=config["index_type"]
        )

        # Load FAISS index
        store.index = faiss.read_index(str(directory / "faiss.index"))

        # Load documents and metadata
        with open(directory / "documents.pkl", "rb") as f:
            store.documents = pickle.load(f)

        with open(directory / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)

        with open(directory / "id_map.json", "r") as f:
            store.id_map = json.load(f)

        logger.info(f"Loaded vector store from {directory} with {len(store.documents)} documents")
        return store

    @property
    def total_documents(self) -> int:
        """Get total number of documents"""
        return len(self.documents)

    def clear(self):
        """Clear all data"""
        self.index = self._create_index(self.index_type)
        self.documents = []
        self.metadata = []
        self.id_map = {}
        logger.info("Cleared vector store")
