from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwenQueryRewriter:
    """Rewrite research questions using a locally cached Qwen model."""

    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        self.model.eval()

    def rewrite(self, query: str) -> str:
        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query rewriting component. "
                    "Rewrite the user's question into ONE concise "
                    "retrieval query. Preserve important technical terms. "
                    "Do not answer the question. "
                    "Do not explain your rewrite. "
                    "Return only the search query."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]

        rewritten = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        if not rewritten:
            raise ValueError("Model returned an empty rewritten query.")

        return rewritten.splitlines()[0].strip()
