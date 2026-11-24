"""Compare src/audio.py (NumPy) with librosa on real recordings.

    python scripts/check_audio_parity.py
"""
import random
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import audio  # noqa: E402
from src.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, TOP_DB  # noqa: E402

files = sorted((ROOT / "data").glob("*/*/*.wav"))
random.Random(0).shuffle(files)
files = files[:400]

basis_lib = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmax=SAMPLE_RATE // 2)
print("mel basis max diff", np.abs(basis_lib - audio.mel_basis(SAMPLE_RATE, N_FFT, N_MELS, SAMPLE_RATE // 2)).max())

worst = {"load": 0, "trim_len": 0, "mel_db": 0}
for f in files:
    a, _ = librosa.load(f, sr=SAMPLE_RATE, mono=True)
    b = audio.read(str(f), SAMPLE_RATE)
    worst["load"] = max(worst["load"], float(np.abs(a - b).max()))
    ta, _ = librosa.effects.trim(a, top_db=TOP_DB)
    tb = audio.trim(b, TOP_DB)
    worst["trim_len"] = max(worst["trim_len"], abs(len(ta) - len(tb)))
    y = np.pad(ta, (0, max(0, 8000 - len(ta))))[:8000]
    ma = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                            n_mels=N_MELS, fmax=SAMPLE_RATE // 2), ref=np.max, top_db=80.0)
    S = audio.melspectrogram(y, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, SAMPLE_RATE // 2)
    mb = audio.power_to_db(S, ref=S.max(), top_db=80.0)
    worst["mel_db"] = max(worst["mel_db"], float(np.abs(ma - mb).max()))

# a 48 kHz recording, like the browser microphone makes
y48 = librosa.resample(librosa.load(files[0], sr=SAMPLE_RATE)[0], orig_sr=SAMPLE_RATE, target_sr=48000)
import soundfile as sf, io  # noqa: E401,E402
buf = io.BytesIO(); sf.write(buf, y48, 48000, format="WAV"); data = buf.getvalue()
a, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE)
b = audio.read(data, SAMPLE_RATE)
n = min(len(a), len(b))
print(f"48 kHz -> 8 kHz: length {len(a)} vs {len(b)}, max diff {np.abs(a[:n] - b[:n]).max():.4f}, "
      f"correlation {np.corrcoef(a[:n], b[:n])[0, 1]:.5f}")
print("worst over", len(files), "files:", worst)
