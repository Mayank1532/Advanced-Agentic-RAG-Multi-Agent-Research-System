import json
from pathlib import Path

from agentic_rag.models import SourceDocument


class DocumentLoader:
    """Load source documents from a local JSON corpus."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[SourceDocument]:
        if not self.path.exists():
            raise FileNotFoundError(f"Document corpus not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError("Document corpus must contain a JSON list.")

        return [SourceDocument.model_validate(record) for record in records]
