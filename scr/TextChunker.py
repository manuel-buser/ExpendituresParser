def chunk_text(text, tokenizer, chunk_size=512):
    """
    Splits the input text into smaller chunks based on the tokenizer,
    ensuring each chunk doesn't exceed 'chunk_size' tokens.
    Returns a list of text chunks.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        token_chunk = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(token_chunk, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
