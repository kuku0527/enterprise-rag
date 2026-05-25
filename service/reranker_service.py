"""Reranker Service for improving retrieval quality"""

from typing import List, Tuple, Optional

from loguru import logger


class RerankerService:
    """Service for reranking retrieved documents"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        Initialize reranker service

        Args:
            model_name: Name of the reranker model
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the reranker model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info(f"Loading reranker model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Reranker model loaded on {self.device}")
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, dict, float]],
        top_n: Optional[int] = None
    ) -> List[Tuple[str, dict, float]]:
        """
        Rerank documents based on query-document relevance

        Args:
            query: Query text
            documents: List of (document, metadata, score) tuples
            top_n: Number of top results to return

        Returns:
            Reranked list of (document, metadata, score) tuples
        """
        if not documents:
            return []

        import torch

        logger.info(f"Reranking {len(documents)} documents")

        # Prepare query-document pairs
        pairs = [(query, doc[0]) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get relevance scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()

        # Create reranked results
        reranked = []
        for i, (doc, meta, original_score) in enumerate(documents):
            reranked.append((
                doc,
                {**meta, "rerank_score": float(scores[i])},
                float(scores[i])
            ))

        # Sort by rerank score
        reranked.sort(key=lambda x: x[2], reverse=True)

        # Apply top_n
        if top_n:
            reranked = reranked[:top_n]

        logger.info(f"Reranking complete, returning top {len(reranked)} results")
        return reranked


class CrossEncoderReranker:
    """Alternative reranker using sentence-transformers CrossEncoder"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded")
        except ImportError:
            raise ImportError(
                "Please install sentence-transformers: pip install sentence-transformers"
            )

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, dict, float]],
        top_n: Optional[int] = None
    ) -> List[Tuple[str, dict, float]]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []

        # Prepare pairs
        pairs = [[query, doc[0]] for doc in documents]

        # Get scores
        scores = self.model.predict(pairs)

        # Create reranked results
        reranked = []
        for i, (doc, meta, original_score) in enumerate(documents):
            reranked.append((
                doc,
                {**meta, "rerank_score": float(scores[i])},
                float(scores[i])
            ))

        # Sort and apply top_n
        reranked.sort(key=lambda x: x[2], reverse=True)
        if top_n:
            reranked = reranked[:top_n]

        return reranked
