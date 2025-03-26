import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from scr.TextChunker import chunk_text


class Falcon7BExtractor:
    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct"):
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Number of CUDA devices:", torch.cuda.device_count())
            current_device = torch.cuda.current_device()
            print("Current CUDA device:", current_device)
            print("CUDA device name:", torch.cuda.get_device_name(current_device))
        else:
            print("No GPU detected. The model will run on CPU.")

        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model_max_length = 2048  # for Falcon 7B



    def extract_transactions_chunked(self, pdf_text: str, chunk_size=512) -> list:
        """
        1) Splits the PDF text into manageable chunks (to avoid exceeding max context length).
        2) For each chunk, calls the model to extract transactions as JSON.
        3) Combines all chunk results into a single list of transactions.
        """

        # --- Step A: Split the text into chunks ---
        text_chunks = chunk_text(pdf_text, self.tokenizer, chunk_size=chunk_size)

        all_transactions = []
        for idx, chunk in enumerate(text_chunks):
            # --- Step B: Build a short prompt for this chunk ---
            prompt = f"""
You are a helpful assistant that extracts banking transactions from text. 
Output only valid JSON, with each transaction having:
- "date" (format DD.MM.YYYY)
- "description"
- "amount" (numeric)

If no transactions are in this chunk, return an empty JSON array: []

Text chunk (do NOT repeat it, just parse it):
{chunk}
"""

            # --- Step C: Tokenize the prompt ---
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length
            )
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # --- Step D: Generate JSON for this chunk ---
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Debug print
            print(f"Chunk {idx+1}/{len(text_chunks)} output:\n{generated_text}\n")

            # --- Step E: Try to parse JSON from the model output ---
            # Some models might produce extra text. Let's do a quick extraction.
            json_str = self.extract_json_str(generated_text)
            if not json_str:
                # If we can't find valid JSON, skip
                continue

            try:
                chunk_transactions = json.loads(json_str)
                # If it's not a list, skip or wrap in list
                if isinstance(chunk_transactions, dict):
                    chunk_transactions = [chunk_transactions]
                # Merge into the global list
                all_transactions.extend(chunk_transactions)
            except json.JSONDecodeError:
                # If it fails, skip
                pass

        return all_transactions

    def extract_json_str(self, text: str) -> str:
        """
        A simple helper that attempts to locate a JSON array in the text.
        If the entire text is valid JSON, returns it. Otherwise tries to find
        the first '[' and last ']' to extract a substring.
        """
        text = text.strip()
        # Quick check if entire text is JSON
        if text.startswith("[") and text.endswith("]"):
            return text

        # Otherwise, find the substring that starts with '[' and ends with ']'
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            return text[start : end + 1]
        return ""



