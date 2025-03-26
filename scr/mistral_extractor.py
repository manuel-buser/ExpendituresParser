import os
import re
import torch
import csv
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM


def chunk_text_by_transaction(pdf_text: str) -> list:
    """
    Splits the PDF text into chunks at transaction boundaries.
    This version uses a regex that looks for lines starting with a date (e.g. "03.02.")
    and assumes each transaction starts with such a line.
    """
    # This pattern matches lines starting with one or two digits, a dot, one or two digits, a dot.
    date_pattern = re.compile(r"^\d{1,2}\.\d{1,2}\.")
    lines = pdf_text.splitlines()
    chunks = []
    current_chunk = []
    for line in lines:
        # Ignore lines that include 'Saldo' as they are not part of the expenditure extraction.
        if "Saldo" in line:
            continue

        # Check if the line starts with a date
        if date_pattern.match(line) and current_chunk:
            # Start a new chunk if we detect a new transaction
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = [line]
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return chunks


def parse_csv_output(text: str) -> list:
    """
    Parses CSV lines from text in the format: Date;Description;Amount
    Returns a list of rows, where each row is a list of values.
    """
    if ";" not in text:
        return []
    rows = []
    f = StringIO(text)
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        if len(row) == 3:
            rows.append(row)
    return rows


class Mistral7BExtractor:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
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
        # Mistral models typically support a 4096-token context.
        self.model_max_length = 4096

    def extract_transactions_by_transaction(self, pdf_text: str) -> list:
        """
        Splits the PDF text into transaction-based chunks and uses the LLM
        to generate CSV lines (Date;Description;Amount) for each chunk.
        Returns a combined list of CSV rows.
        """
        # Step A: Chunk text logically by transaction
        chunks = chunk_text_by_transaction(pdf_text)
        all_rows = []
        for idx, chunk in enumerate(chunks):
            # Step B: Build a detailed prompt with extra context.
            prompt = f"""
You are a helpful assistant that extracts banking transactions from text.
The following context applies:
- Any line containing the word "Saldo" is not part of an expenditure and should be ignored.
- "Valuta" indicates the actual transaction date; if available, use that as the Date. If not, use the first date you see.
- "Belastung" means money is spent. Prefix the amount with a "-" (negative).
- "Gutschrift" means money is credited. Prefix the amount with a "+" (positive).
- The description should capture the merchant or purpose (for example, "BASSO BASEL" or "Coop-4074 Basel Elsasser Basel") and ignore details like card numbers.
Return only CSV lines in the following format (without any header):
Date;Description;Amount
For example:
03.02.2025;BASSO BASEL;-100.00

Do NOT echo the text chunk. Only output CSV lines.

Text chunk:
{chunk}
"""
            # Step C: Tokenize and generate
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_max_length
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Chunk {idx + 1}/{len(chunks)} output:\n{generated_text}\n")

            # Step D: Parse CSV output
            rows = parse_csv_output(generated_text)
            all_rows.extend(rows)
        return all_rows
