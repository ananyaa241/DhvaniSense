AI-Generated Voice Detection (Multi-Language)

DhvaniSense is an API-based system that detects whether a given voice sample is AI-generated or Human, across multiple languages, using audio signal processing and a lightweight machine learning model enhanced with a custom Micro-Prosody Entropy Score (MPES).


📘 README.md — DhvaniSense
AI-Generated Voice Detection API

(End-to-End Setup → Final URL Generation)

This project provides a REST API that detects whether a given voice sample is AI-generated or Human.
Follow the steps below exactly in order to run the project and generate the final API URL.

✅ Step 0 — Prerequisites

Ensure the following are installed on your system:

1️⃣ Python

Version 3.9 or above

Verify:

python --version

2️⃣ FFmpeg (Required for MP3 decoding)

Download from: https://www.gyan.dev/ffmpeg/builds/

Extract and add the bin/ folder to system PATH

Verify:

ffmpeg -version

📦 Step 1 — Install Python Dependencies

Open the project folder in VS Code and open a terminal.

From the project root:

pip install -r requirements.txt


This installs:

FastAPI

Uvicorn

PyTorch

Librosa

NumPy

SciPy

Pydub

🎧 Step 2 — Prepare the Training Dataset

Ensure training audio is placed as follows:

training/dataset/
├── human/
│   ├── human1.mp3
│   ├── human2.mp3
│   └── ...
└── ai/
    ├── ai1.mp3
    ├── ai2.mp3
    └── ...


Rules:

MP3 format only

One speaker per file

Avoid empty or noisy audio

🧠 Step 3 — Train the Model

From the project root:

cd training
python train.py
cd ..


What this does:

Loads training audio

Extracts MFCC features

Trains a neural network

Saves trained weights to:

api/model_weights.pth


⚠️ This step is required only once, unless you add more data or modify training logic.

🚀 Step 4 — Start the API Server

From the project root:

cd api
uvicorn app:app


If successful, you will see:

Uvicorn running on http://127.0.0.1:8000

🌐 Step 5 — Final Generated URLs

Once the server is running, the following URLs are available:

🔹 Base API URL
http://127.0.0.1:8000

🔹 API Endpoint URL (for requests)
http://127.0.0.1:8000/api/voice-detection

🔹 Interactive API Documentation (Swagger UI)
http://127.0.0.1:8000/docs


👉 This /docs URL is the final URL used for testing and validation.

🔐 Step 6 — API Authentication

All requests must include the following header:

x-api-key: sk_dhvanisense_2026


Requests without this key will be rejected.

📡 Step 7 — Test the API (Example)
Request Body
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_MP3>"
}

Response Example
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.63,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}

🔁 Restarting After Shutdown

If the laptop is restarted:

cd api
uvicorn app:app


No retraining is required if model_weights.pth exists.

🧠 Summary (One-Line)

After training the model and starting the FastAPI server, the final usable URL is
http://127.0.0.1:8000/docs
