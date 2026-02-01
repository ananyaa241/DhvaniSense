import os
import sys

# 🔥 Fix import path (IMPORTANT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
import librosa
import numpy as np

from api.model import VoiceClassifier
from api.audio_utils import extract_mfcc

X = []
y = []

DATASET_DIR = os.path.join(PROJECT_ROOT, "training", "dataset")

for label, folder in [(0, "human"), (1, "ai")]:
    folder_path = os.path.join(DATASET_DIR, folder)

    for file in os.listdir(folder_path):
        if not file.lower().endswith(".mp3"):
            continue

        audio, sr = librosa.load(os.path.join(folder_path, file), sr=16000)
        features = extract_mfcc(audio, sr)

        X.append(features)
        y.append(label)

X = torch.tensor(X).float()
y = torch.tensor(y).float().unsqueeze(1)

model = VoiceClassifier()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/30 - Loss: {loss.item():.4f}")

MODEL_PATH = os.path.join(PROJECT_ROOT, "api", "model_weights.pth")
torch.save(model.state_dict(), MODEL_PATH)

print("✅ Model training completed and saved.")
