from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from my_lib.summarize import load_model_and_tokenizer, summarize_text
from typing import Tuple
import torch

app = FastAPI()

# Global variables for model and tokenizer
tokenizer = None
model = None

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    global tokenizer, model
    tokenizer, model = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Summarization API. Use /summarize to summarize text."}

@app.post("/summarize")
async def summarize_api(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Summarize large text by processing paragraph by paragraph
        summary = summarize_text(request.text, tokenizer, model)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")
