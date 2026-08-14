class ReviewerAgent:
    """Agent responsible for basic evidence-grounding checks."""

    def review(
        self,
        draft: str,
        evidence: list[str],
    ) -> tuple[str, bool]:
        if not draft.strip():
            return "Rejected: draft is empty.", False

        if not evidence:
            return "Rejected: no supporting evidence.", False

        required_sections = (
            "# Research Report",
            "## Question",
            "## Analysis",
        )

        missing_sections = [
            section
            for section in required_sections
            if section not in draft
        ]

        if missing_sections:
            return (
                "Rejected: missing report sections: "
                + ", ".join(missing_sections),
                False,
            )

        evidence_matches = sum(
            evidence_item[:40].strip() in draft
            for evidence_item in evidence
            if evidence_item[:40].strip()
        )

        if evidence_matches == 0:
            return (
                "Rejected: draft does not contain "
                "recognizable retrieved evidence.",
                False,
            )

        return (
            "Approved: report contains retrieved evidence "
            "and required report sections.",
            True,
        )
