import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads the entire PDF and returns its text as a string.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
