from agentic_rag.models import DocumentChunk, SourceDocument


class TextChunker:
    """Split source documents into deterministic overlapping chunks."""

    def __init__(self, chunk_size: int = 80, overlap: int = 15) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: SourceDocument) -> list[DocumentChunk]:
        words = document.text.split()

        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: list[DocumentChunk] = []

        for chunk_index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + self.chunk_size]

            if not chunk_words:
                continue

            chunks.append(
                DocumentChunk(
                    id=f"{document.id}-{chunk_index}",
                    text=" ".join(chunk_words),
                    source=document.source,
                    title=document.title,
                    category=document.category,
                    chunk_index=chunk_index,
                    metadata={
                        **document.metadata,
                        "document_id": document.id,
                    },
                )
            )

            if start + self.chunk_size >= len(words):
                break

        return chunks

    def chunk_documents(
        self,
        documents: list[SourceDocument],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks
