"""Tests for RAG System"""

import pytest
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from loader.document_loader import Document, TextLoader
from parser.text_parser import TextParser, TextChunk
from vectorstore.faiss_store import FAISSVectorStore


class TestTextParser:
    """Test text parsing functionality"""

    def test_fixed_size_chunking(self):
        """Test fixed-size text chunking"""
        parser = TextParser(chunk_size=100, chunk_overlap=20)
        text = "这是一段测试文本。" * 20
        chunks = parser.parse(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # Allow some flexibility

    def test_empty_text(self):
        """Test parsing empty text"""
        parser = TextParser()
        chunks = parser.parse("")
        assert len(chunks) == 0

    def test_metadata_preservation(self):
        """Test that metadata is preserved"""
        parser = TextParser(chunk_size=50, chunk_overlap=10)
        metadata = {"source": "test.txt", "page": 1}
        chunks = parser.parse("测试文本" * 10, metadata=metadata)

        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["page"] == 1


class TestDocumentLoader:
    """Test document loading functionality"""

    def test_text_loader(self):
        """Test text file loading"""
        # Create a temporary text file
        test_file = Path("test_temp.txt")
        test_file.write_text("这是测试内容", encoding="utf-8")

        try:
            loader = TextLoader()
            docs = loader.load(str(test_file))

            assert len(docs) == 1
            assert docs[0].content == "这是测试内容"
            assert docs[0].metadata["file_type"] == "txt"
        finally:
            test_file.unlink()


class TestFAISSVectorStore:
    """Test FAISS vector store functionality"""

    def test_add_and_search(self):
        """Test adding documents and searching"""
        store = FAISSVectorStore(dimension=128)

        # Create random embeddings
        embeddings = np.random.randn(5, 128).astype(np.float32)
        documents = ["文档1", "文档2", "文档3", "文档4", "文档5"]
        metadata = [{"id": i} for i in range(5)]

        store.add_documents(embeddings, documents, metadata)

        assert store.total_documents == 5

        # Search
        query = np.random.randn(128).astype(np.float32)
        results = store.search(query, top_k=3)

        assert len(results) <= 3
        assert all(len(r) == 3 for r in results)

    def test_save_and_load(self):
        """Test saving and loading vector store"""
        store = FAISSVectorStore(dimension=128)

        embeddings = np.random.randn(3, 128).astype(np.float32)
        documents = ["测试文档1", "测试文档2", "测试文档3"]
        store.add_documents(embeddings, documents)

        # Save
        test_dir = Path("test_vector_store")
        store.save(str(test_dir))

        # Load
        loaded_store = FAISSVectorStore.load(str(test_dir))

        assert loaded_store.total_documents == 3
        assert loaded_store.documents == documents

        # Cleanup
        import shutil
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
