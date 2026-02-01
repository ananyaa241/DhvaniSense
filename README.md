AI-Generated Voice Detection (Multi-Language)

DhvaniSense is an API-based system that detects whether a given voice sample is AI-generated or Human, across multiple languages, using audio signal processing and a lightweight machine learning model enhanced with a custom Micro-Prosody Entropy Score (MPES).



📂 Project Structure
DhvaniSense/

├── api/

│   ├── app.py
                            # FastAPI application
│   ├── model.py  
                            # Neural network model definition
│   ├── audio_utils.py      
                            # Audio loading & feature extraction
│   └── mpes.py             # Micro-Prosody Entropy Score logic
│
├── training/

│   ├── train.py          
                            # Model training script
│   └── dataset/

│       ├── human/      
                            # Human voice MP3 files
│       └── ai/             
                            # AI-generated voice MP3 files
│

├── requirements.txt


└── .gitignore




⚙️ Prerequisites
1️⃣ Python

Python 3.9 or above

2️⃣ FFmpeg (Required)

FFmpeg is required for decoding MP3 audio files.

Download from: https://www.gyan.dev/ffmpeg/builds/

Add the bin/ directory to the system PATH

Verify installation:

ffmpeg -version




📦 Install Dependencies

From the project root directory:

pip install -r requirements.txt


🎧 Preparing the Training Dataset
Dataset Format

Training data must be organized as follows:

training/dataset/
├── human/
│   ├── human1.mp3
│   ├── human2.mp3
│   └── ...
└── ai/
    ├── ai1.mp3
    ├── ai2.mp3
    └── ...

Dataset Guidelines

Audio format: MP3

One speaker per file

Avoid long silences or heavy background noise

Balanced human and AI samples are recommended



🧠 Training the Model

Run the training script from the project root:

cd training
python train.py

Training Pipeline

MP3 audio is resampled to 16 kHz

MFCC features are extracted

Labels:

0 → Human

1 → AI-generated

Model is trained using supervised learning

Trained weights are saved to:

api/model_weights.pth




🔁 Improving the Model

The model can be improved by:

Adding more MP3 samples to human/ and ai/

Adjusting training parameters in training/train.py

Number of epochs

Learning rate

Network size

Modifying the MPES logic in api/mpes.py to experiment with:

Pitch entropy thresholds

Additional micro-prosodic features

Retraining is required after any dataset or model change.




🚀 Running the API

Start the API server from the project root:

cd api
uvicorn app:app --reload


The API will be available at:

http://127.0.0.1:8000


Interactive API documentation (Swagger UI):

http://127.0.0.1:8000/docs





🔐 API Authentication

All API requests must include the following header:

x-api-key: sk_dhvanisense_2026


Requests without a valid API key will be rejected.




📡 API Endpoint
POST /api/voice-detection
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



🧪 Notes

Do not commit:

venv/

model_weights.pth

FFmpeg binaries

Keep the API request/response format unchanged

The system is designed to be language-agnostic across supported languages





🧠 Core Approach

The system combines:

MFCC-based neural network classification

Micro-Prosody Entropy Score (MPES) to detect unnaturally smooth pitch behavior commonly observed in AI-generated speech.
