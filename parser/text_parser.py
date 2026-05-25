"""Text Parsing and Chunking Strategies"""

from typing import List, Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    metadata: dict
    index: int

    def __len__(self):
        return len(self.content)


class TextParser:
    """Simple fixed-size text chunking with overlap"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Parse text into fixed-size chunks"""
        metadata = metadata or {}
        chunks = []

        # Clean text
        text = text.strip()
        if not text:
            return chunks

        # Split into chunks
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to find a natural break point
            if end < len(text):
                # Look for sentence boundary
                for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
                    pos = text.rfind(sep, start + self.chunk_size // 2, end)
                    if pos != -1:
                        end = pos + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end
                }
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    index=chunk_index
                ))
                chunk_index += 1

            # Move start position
            start = end - self.chunk_overlap if end < len(text) else len(text)

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


class SentenceParser:
    """Sentence-level text chunking"""

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Handle Chinese and English sentence endings
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def parse(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Parse text into sentence-based chunks"""
        metadata = metadata or {}
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = "".join(current_chunk)
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "sentence_count": len(current_chunk)
                }
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    index=chunk_index
                ))
                chunk_index += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

        # Handle remaining sentences
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "sentence_count": len(current_chunk)
            }
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                index=chunk_index
            ))

        logger.info(f"Split text into {len(chunks)} sentence-based chunks")
        return chunks


class SemanticParser:
    """Semantic-aware text chunking using embeddings similarity"""

    def __init__(self, similarity_threshold: float = 0.5, max_chunk_size: int = 1000):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def parse(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Parse text into semantically coherent chunks"""
        # For simplicity, use paragraph-based splitting
        # In production, use embedding similarity to merge related paragraphs
        metadata = metadata or {}
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            if current_length + len(para) > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "paragraph_count": len(current_chunk)
                }
                chunks.append(TextChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    index=chunk_index
                ))
                chunk_index += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += len(para)

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "paragraph_count": len(current_chunk)
            }
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                index=chunk_index
            ))

        logger.info(f"Split text into {len(chunks)} semantic chunks")
        return chunks
