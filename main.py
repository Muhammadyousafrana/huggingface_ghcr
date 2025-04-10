from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from my_lib.summarize import summarize, detect_language_code

# Initialize FastAPI app
app = FastAPI()


# Define input request model
class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to the Transcription API. Use /summarize to upload and process files."}


# API Endpoint for Summarization
@app.post("/summarize")
async def summarize_text(request: TextRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # # Detect language
    # src_lang = detect_language_code(text)
    # tgt_lang = "en_XX" if src_lang != "en_XX" else "en_XX"  # default to English summary

    # Generate summary
    summary = summarize(text)
    return {"summary": summary}
