from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Load mT5 model and tokenizer
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Supported languages (optional filter)
SUPPORTED_LANGUAGES = [
    "bn", "en", "es", "fr", "hi", "ru", "ur", "zh",  # etc.
]

# Language detection
def detect_language_code(text):
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"

# Summarization function
def summarize(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=128,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
