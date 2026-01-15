from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# -------------------- APP INIT --------------------
app = FastAPI(
    title="Scam Detection API",
    description="Multimodal scam detection using DistilBERT",
    version="1.0.0"
)

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "scam_model")

# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- LOAD MODEL (ONCE) --------------------
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -------------------- REQUEST SCHEMA --------------------
class PredictRequest(BaseModel):
    text: str
    modality: str  # sms / whatsapp / call / audio

# -------------------- HEALTH CHECK --------------------
@app.get("/")
def health():
    return {"status": "Scam Detection API running"}

# -------------------- PREDICTION --------------------
@app.post("/predict")
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        scam_prob = probs[0][1].item()

    # -------------------- RISK LEVEL LOGIC --------------------
    if scam_prob >= 0.85:
        risk_level = "HIGH"
    elif scam_prob >= 0.70:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "text": req.text[:100],  # preview
        "modality": req.modality,
        "is_scam": int(scam_prob > 0.5),
        "confidence": round(scam_prob, 3),
        "risk_level": risk_level
    }
