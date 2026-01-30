import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Prompt Safety API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Load once at startup (path from env MODEL_DIR or config default)
tokenizer = AutoTokenizer.from_pretrained(settings.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(settings.model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {0: "safe", 1: "unsafe"}

class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=settings.max_prompt_length)

@app.post("/predict")
@limiter.limit(settings.predict_rate_limit)
def predict(request: Request, req: PredictRequest):
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # [1,2]
        probs = F.softmax(logits, dim=-1)[0]

    safe_prob = float(probs[0].item())
    unsafe_prob = float(probs[1].item())
    pred_id = 0 if safe_prob >= unsafe_prob else 1

    return {
        "prompt":req.prompt,
        "label": id2label[pred_id],
        "safe_probability": safe_prob,
        "unsafe_probability": unsafe_prob,
    }