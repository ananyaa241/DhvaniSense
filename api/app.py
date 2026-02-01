import os
import sys
import uvicorn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from fastapi import FastAPI, Header
from pydantic import BaseModel
import torch

from audio_utils import decode_base64_mp3, extract_mfcc
from model import VoiceClassifier
from mpes import micro_prosody_entropy



API_KEY = "sk_dhvanisense_2026"

app = FastAPI()

model = VoiceClassifier()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        return {"status": "error", "message": "Invalid API key"}

    if req.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format"}

    try:
        y, sr = decode_base64_mp3(req.audioBase64)
        mfcc = extract_mfcc(y, sr)

        x = torch.tensor(mfcc).float().unsqueeze(0)
        with torch.no_grad():
            score = model(x).item()

        mpes = micro_prosody_entropy(y, sr)

        # 🔥 MPES calibration (unique logic)
        if mpes < 0.8:
            score = min(score + 0.15, 1.0)
        elif mpes > 1.2:
            score = max(score - 0.15, 0.0)

        classification = "AI_GENERATED" if score > 0.5 else "HUMAN"

        explanation = (
            "Unnatural pitch consistency and robotic speech patterns detected"
            if classification == "AI_GENERATED"
            else
            "Natural pitch variation and human speech characteristics detected"
        )

        return {
            "status": "success",
            "language": req.language,
            "classification": classification,
            "confidenceScore": round(score, 2),
            "explanation": explanation
        }

    except Exception:
        return {"status": "error", "message": "Malformed request"}
