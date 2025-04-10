from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

app = FastAPI()

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# Load model and tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# Request body schema
class TextRequest(BaseModel):
    text: str

# Clean the text before processing
def clean_text(text: str) -> str:
    # Remove control characters and normalize whitespace
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)  # Remove control chars
    text = re.sub(r"\n+", " ", text)              # Replace newlines with space
    text = re.sub(r"\s+", " ", text)              # Normalize whitespace
    return text.strip()

# Summarize the text
def summarize_text(text: str, tokenizer, model):
    device = model.device
    # Clean the text before tokenizing
    cleaned = clean_text(text)
    
    # Tokenize the input text and ensure it's within the max length (2000 tokens)
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=2000
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=524,
            min_length=190,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# FastAPI endpoint
@app.post("/summarize/")
async def summarize(request: TextRequest):
    try:
        text = request.text
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        summary = summarize_text(text, tokenizer, model)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
