"""Main RAG Service orchestrating the complete pipeline"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from loader import DocumentLoader
from parser import TextParser
from vectorstore import FAISSVectorStore
from service.embedding_service import EmbeddingService
from service.reranker_service import RerankerService


class RAGService:
    """Complete RAG service with document loading, parsing, indexing, and retrieval"""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_reranker: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize RAG service

        Args:
            embedding_model: Name of embedding model
            reranker_model: Name of reranker model
            chunk_size: Text chunk size
            chunk_overlap: Overlap between chunks
            use_reranker: Whether to use reranking
            device: Device for models
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_reranker = use_reranker

        # Initialize components
        logger.info("Initializing RAG service components...")

        self.text_parser = TextParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            device=device
        )

        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_service.dimension
        )

        if use_reranker:
            self.reranker = RerankerService(
                model_name=reranker_model,
                device=device
            )
        else:
            self.reranker = None

        logger.info("RAG service initialized successfully")

    def ingest_documents(self, path: str, is_directory: bool = False) -> Dict[str, Any]:
        """
        Ingest documents into the vector store

        Args:
            path: Path to file or directory
            is_directory: Whether path is a directory

        Returns:
            Ingestion statistics
        """
        logger.info(f"Ingesting documents from: {path}")

        # Load documents
        if is_directory:
            documents = DocumentLoader.load_directory(path)
        else:
            documents = DocumentLoader.load_file(path)

        if not documents:
            logger.warning("No documents loaded")
            return {"total_documents": 0, "total_chunks": 0}

        # Parse into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_parser.parse(
                doc.content,
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        # Add to vector store
        metadata = [chunk.metadata for chunk in all_chunks]
        self.vector_store.add_documents(embeddings, texts, metadata)

        stats = {
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
            "embedding_dimension": embeddings.shape[1]
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank_top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Query the RAG system

        Args:
            question: User question
            top_k: Number of documents to retrieve
            rerank_top_n: Number of documents after reranking

        Returns:
            Query results with answer and sources
        """
        logger.info(f"Processing query: {question}")

        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(question)

        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        if not results:
            return {
                "answer": "未找到相关文档",
                "sources": [],
                "confidence": 0.0
            }

        # Apply reranking if enabled
        if self.use_reranker and self.reranker:
            results = self.reranker.rerank(
                question,
                results,
                top_n=rerank_top_n
            )

        # Prepare context
        context = self._prepare_context(results)

        # Generate answer (using LLM)
        answer = self._generate_answer(question, context)

        # Prepare response
        sources = []
        for doc, meta, score in results:
            sources.append({
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "metadata": meta,
                "score": score
            })

        response = {
            "answer": answer,
            "sources": sources,
            "confidence": results[0][2] if results else 0.0
        }

        logger.info(f"Query processed, found {len(sources)} relevant sources")
        return response

    def _prepare_context(self, results: List[tuple]) -> str:
        """Prepare context from search results"""
        context_parts = []
        for i, (doc, meta, score) in enumerate(results, 1):
            source = meta.get("source", "unknown")
            context_parts.append(f"[文档{i}] (来源: {source}, 相关度: {score:.2f})\n{doc}")

        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM

        This is a placeholder. In production, integrate with OpenAI, local LLM, etc.
        """
        # For demo purposes, return a simple response
        # In production, use LLM API to generate answer
        prompt = f"""基于以下上下文信息回答问题。如果无法从上下文中找到答案，请说明。

上下文：
{context}

问题：{question}

回答："""

        # TODO: Integrate with actual LLM
        # For now, return a formatted response
        return f"基于检索到的 {context.count('[文档]')} 个相关文档，以下是关于「{question}」的回答：\n\n" \
               f"根据检索到的文档内容，找到了以下相关信息：\n" \
               f"{context[:500]}...\n\n" \
               f"注：这是一个演示回答。在生产环境中，请配置 LLM API 以生成更准确的回答。"

    def save(self, directory: str):
        """Save RAG service state"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(str(directory / "vector_store"))
        logger.info(f"Saved RAG service to {directory}")

    @classmethod
    def load(cls, directory: str, **kwargs) -> "RAGService":
        """Load RAG service from directory"""
        directory = Path(directory)

        service = cls(**kwargs)
        service.vector_store = FAISSVectorStore.load(str(directory / "vector_store"))

        logger.info(f"Loaded RAG service from {directory}")
        return service
