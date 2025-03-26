from pdf_extractor import extract_text_from_pdf
from falcon_extractor import Falcon7BExtractor

def main():
    # 1) Read the PDF text
    pdf_path = "../data/Kontoauszug.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    print("Extracted PDF text snippet:")
    print(pdf_text[:300])

    # 2) Initialize the extractor
    extractor = Falcon7BExtractor()

    # 3) Use chunked extraction to get transactions
    all_transactions = extractor.extract_transactions_chunked(pdf_text, chunk_size=512)
    print("Final combined transactions:")
    print(all_transactions)

if __name__ == "__main__":
    main()
