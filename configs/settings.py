"""RAG System Configuration"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)


class LLMConfig:
    """LLM Configuration"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


class EmbeddingConfig:
    """Embedding Configuration"""
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
    BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


class ChunkingConfig:
    """Document Chunking Configuration"""
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    SEPARATOR = "\n"


class RetrievalConfig:
    """Retrieval Configuration"""
    TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))


class APIConfig:
    """API Configuration"""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG = os.getenv("API_DEBUG", "true").lower() == "true"
