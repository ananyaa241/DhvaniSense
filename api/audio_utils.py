import os

# 🔥 FORCE FFMPEG PATH FIRST (CRITICAL)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg", "bin", "ffmpeg.exe")
FFPROBE_PATH = os.path.join(BASE_DIR, "ffmpeg", "bin", "ffprobe.exe")

os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

from pydub import AudioSegment
import base64
import io
import librosa
import numpy as np

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH


def decode_base64_mp3(audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)

    y, sr = librosa.load(buf, sr=16000)
    return y, sr


def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)
