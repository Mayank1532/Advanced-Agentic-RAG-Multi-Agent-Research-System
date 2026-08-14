class WriterAgent:
    """Agent responsible for producing the research report."""

    def write(self, question: str, analysis: str) -> str:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not analysis.strip():
            raise ValueError("Analysis cannot be empty.")

        return (
            "# Research Report\n\n"
            f"## Question\n\n{question}\n\n"
            "## Analysis\n\n"
            f"{analysis}\n"
        )
