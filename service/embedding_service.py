"""Embedding Service using Sentence Transformers"""

from typing import List, Optional

import numpy as np
from loguru import logger


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        """
        Initialize embedding service

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            numpy array of embeddings (n x dimension)
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        return embedding[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingService:
    """OpenAI API-based embedding service"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding service

        Args:
            api_key: OpenAI API key
            base_url: API base URL (for compatible APIs)
            model: Embedding model name
        """
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info(f"Initialized OpenAI embedding service with model: {self.model}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[query]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        # Common dimensions for OpenAI models
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dim_map.get(self.model, 1536)
