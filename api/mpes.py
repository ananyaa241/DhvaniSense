import librosa
import numpy as np
from scipy.stats import entropy

def micro_prosody_entropy(y, sr):
    pitch, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    pitch = pitch[~np.isnan(pitch)]
    if len(pitch) < 10:
        return 0.0

    delta = np.diff(pitch)
    hist, _ = np.histogram(delta, bins=20, density=True)
    hist += 1e-9

    return entropy(hist, base=2)
