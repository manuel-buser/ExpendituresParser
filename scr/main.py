from pdf_extractor import extract_text_from_pdf
from mistral_extractor import Mistral7BExtractor

def main():
    pdf_path = "../data/Kontoauszug.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    print("Extracted PDF text snippet:")
    print(pdf_text[:300])

    extractor = Mistral7BExtractor()
    transactions = extractor.extract_transactions_by_transaction(pdf_text)
    print("Final combined CSV rows:")
    for row in transactions:
        print(row)

if __name__ == "__main__":
    main()
