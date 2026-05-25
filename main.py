"""
Enterprise RAG System - Main Entry Point

Features:
- Document loading (PDF, DOCX, TXT)
- Text chunking with multiple strategies
- Vector embedding with FAISS
- Reranking for improved retrieval
- REST API for querying
"""

import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from service import RAGService
from configs.settings import LLMConfig, EmbeddingConfig, ChunkingConfig, RetrievalConfig, APIConfig

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG System",
    description="检索增强生成系统 - 支持文档问答",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG service instance
rag_service: Optional[RAGService] = None


class QueryRequest(BaseModel):
    """Query request model"""
    question: str
    top_k: int = 5
    rerank_top_n: int = 3


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: list
    confidence: float


class IngestRequest(BaseModel):
    """Ingest request model"""
    path: str
    is_directory: bool = False


class IngestResponse(BaseModel):
    """Ingest response model"""
    total_documents: int
    total_chunks: int
    embedding_dimension: int


@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup"""
    global rag_service

    logger.info("Starting Enterprise RAG System...")

    try:
        rag_service = RAGService(
            embedding_model=EmbeddingConfig.MODEL_NAME,
            chunk_size=ChunkingConfig.CHUNK_SIZE,
            chunk_overlap=ChunkingConfig.CHUNK_OVERLAP,
            use_reranker=RetrievalConfig.USE_RERANKER,
            device=EmbeddingConfig.DEVICE
        )
        logger.info("RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Enterprise RAG System",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system

    Args:
        request: Query request with question and parameters

    Returns:
        QueryResponse with answer and sources
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store

    Args:
        request: Ingest request with path and options

    Returns:
        IngestResponse with statistics
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        stats = rag_service.ingest_documents(
            path=request.path,
            is_directory=request.is_directory
        )

        return IngestResponse(**stats)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a file

    Args:
        file: Uploaded file

    Returns:
        Ingestion statistics
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        # Save uploaded file temporarily
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest the file
        stats = rag_service.ingest_documents(str(file_path))

        # Clean up
        file_path.unlink()

        return {
            "message": f"File {file.filename} ingested successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get vector store statistics"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    return {
        "total_documents": rag_service.vector_store.total_documents,
        "embedding_dimension": rag_service.embedding_service.dimension,
        "use_reranker": rag_service.use_reranker
    }


@app.post("/save")
async def save_index(directory: str = "vector_store"):
    """Save the current vector store"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        rag_service.save(directory)
        return {"message": f"Vector store saved to {directory}"}
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_index(directory: str = "vector_store"):
    """Load a saved vector store"""
    global rag_service

    try:
        rag_service = RAGService.load(directory)
        return {"message": f"Vector store loaded from {directory}"}
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_cli():
    """Run interactive CLI mode"""
    print("\n" + "="*60)
    print("Enterprise RAG System - 交互模式")
    print("="*60)

    # Initialize RAG service
    try:
        service = RAGService(
            embedding_model=EmbeddingConfig.MODEL_NAME,
            chunk_size=ChunkingConfig.CHUNK_SIZE,
            chunk_overlap=ChunkingConfig.CHUNK_OVERLAP,
            use_reranker=RetrievalConfig.USE_RERANKER,
            device=EmbeddingConfig.DEVICE
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # Ingest documents
    data_dir = Path("data")
    if data_dir.exists():
        print(f"\n正在加载文档目录: {data_dir}")
        stats = service.ingest_documents(str(data_dir), is_directory=True)
        print(f"加载完成: {stats['total_documents']} 个文档, {stats['total_chunks']} 个文本块")
    else:
        print(f"\n文档目录不存在: {data_dir}")
        print("请将文档放入 data/ 目录后重试")

    # Interactive query loop
    print("\n" + "-"*60)
    print("输入问题进行查询，输入 'quit' 退出")
    print("-"*60 + "\n")

    while True:
        try:
            question = input("\n请输入问题: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n再见！")
                break

            if not question:
                continue

            print("\n正在查询...")
            result = service.query(question)

            print("\n" + "="*60)
            print("回答:")
            print("="*60)
            print(result["answer"])

            print("\n" + "-"*60)
            print("相关文档:")
            print("-"*60)
            for i, source in enumerate(result["sources"], 1):
                print(f"\n[{i}] 相关度: {source['score']:.2f}")
                print(f"来源: {source['metadata'].get('source', 'unknown')}")
                print(f"内容: {source['content'][:100]}...")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n查询出错: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise RAG System")
    parser.add_argument("--mode", choices=["api", "cli"], default="cli",
                       help="运行模式: api (REST API) 或 cli (命令行)")
    parser.add_argument("--host", default=APIConfig.HOST, help="API host")
    parser.add_argument("--port", type=int, default=APIConfig.PORT, help="API port")

    args = parser.parse_args()

    if args.mode == "api":
        print(f"\n启动 RAG API 服务: http://{args.host}:{args.port}")
        print(f"API 文档: http://{args.host}:{args.port}/docs\n")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        run_cli()
