class AnalystAgent:
    """Agent responsible for synthesizing retrieved evidence."""

    def analyze(self, evidence: list[str]) -> str:
        if not evidence:
            raise ValueError("Cannot analyze empty evidence.")

        sections = [
            f"Evidence {index}: {text}"
            for index, text in enumerate(evidence, start=1)
        ]

        return "\n\n".join(sections)
