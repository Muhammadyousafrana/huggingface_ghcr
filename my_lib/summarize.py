from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model

def clean_text(text: str) -> str:
    # Remove control characters and normalize whitespace
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)  # Remove control chars
    text = re.sub(r"\s+", " ", text)              # Normalize whitespace
    text = re.sub(r"\n+", " ", text)              # Replace newlines with space
    return text.strip()

def summarize_text(text: str, tokenizer, model):
    device = model.device
    cleaned = clean_text(text)

    # Tokenize the input text and ensure it's within the max length (2000 tokens)
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=2000
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def split_text_into_chunks(text: str, tokenizer, max_tokens=500):
    # Tokenize the text into tokens and split into chunks
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return chunks

def summarize_large_text(text: str, tokenizer, model):
    # Split the text into chunks of 500 tokens
    chunks = split_text_into_chunks(text, tokenizer, max_tokens=500)
    
    # List to store the summaries of each chunk
    chunk_summaries = []
    
    for chunk in chunks:
        # Decode token chunk to text and summarize
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        if chunk_text.strip():  # Avoid empty chunks
            summary = summarize_text(chunk_text, tokenizer, model)
            chunk_summaries.append(summary)
    
    # Combine all chunk summaries into one summary
    combined_summary = " ".join(chunk_summaries)
    return combined_summary
