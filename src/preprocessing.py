"""Step 2: data processing.

Every recording goes through the same steps, whether it comes from the
training set, a bulk upload or a single prediction:

1. load as mono and resample to 8 kHz
2. cut the silence at the start and end
3. scale so the loudest sample is 1 (quiet and loud speakers look alike)
4. pad with silence or cut to exactly one second: a short word is centred,
   a long recording keeps its loudest second
5. turn it into a log-mel spectrogram: 40 frequency bands x 101 time steps,
   scaled to the range 0..1

The model sees the spectrogram as a small one-channel image.

The maths is in src/audio.py (NumPy and SciPy only), which matches librosa
to within 0.00003 dB (scripts/check_audio_parity.py). librosa is only used
for the chart features in describe(), which run offline.
"""
import re
from pathlib import Path

import numpy as np

from src import audio

from src.config import (CLIP_SECONDS, HOP_LENGTH, LABELS, N_FFT, N_MELS,
                        NOISE_DIR, SAMPLE_RATE, TOP_DB)

CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


class AudioError(ValueError):
    """The file can't be used: unreadable, silent, or too long."""


def load_audio(source):
    """Load a path, bytes or file object as a mono 8 kHz float32 array."""
    try:
        y = audio.read(source, SAMPLE_RATE)
    except Exception as exc:  # soundfile raises many different errors
        raise AudioError(f"could not read audio ({exc.__class__.__name__})") from exc
    if y.size == 0:
        raise AudioError("the file has no audio")
    return y.astype(np.float32)


def trim_silence(y):
    return audio.trim(y, TOP_DB)


def normalise(y):
    peak = np.max(np.abs(y))
    if peak < 1e-4:
        raise AudioError("the recording is silent")
    return y / peak


def fix_length(y, n=CLIP_SAMPLES):
    """Exactly n samples. A longer clip keeps its loudest second, which is the
    word rather than the background noise around it (a microphone recording
    in a noisy room is not trimmed much). A shorter clip is centred with
    silence on both sides."""
    if len(y) > n:
        energy = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
        start = int(np.argmax(energy[n:] - energy[:-n]))
        return y[start:start + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def clean(y, max_seconds=3.0):
    """Steps 2 to 4. Returns the fixed-length clip and the spoken length."""
    y = trim_silence(y)
    spoken = len(y) / SAMPLE_RATE
    if spoken < 0.08:
        raise AudioError("no speech found")
    if spoken > max_seconds:
        raise AudioError(f"too long ({spoken:.1f} s); record one digit per file")
    return fix_length(normalise(y)), spoken


def log_mel(y):
    """Step 5: a (40, 101) log-mel spectrogram scaled to 0..1."""
    mel = audio.melspectrogram(y, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, SAMPLE_RATE // 2)
    db = audio.power_to_db(mel, ref=mel.max(), top_db=80.0)
    return ((db + 80.0) / 80.0).astype(np.float32)


def prepare(source):
    """Any audio source to a model input of shape (40, 101, 1)."""
    y, _ = clean(load_audio(source))
    return log_mel(y)[..., np.newaxis]


def prepare_many(sources):
    """Prepare a list of files. Returns the inputs and the indexes that worked."""
    xs, ok = [], []
    for i, s in enumerate(sources):
        try:
            xs.append(prepare(s))
            ok.append(i)
        except AudioError:
            continue
    if not xs:
        return np.zeros((0, N_MELS, CLIP_SAMPLES // HOP_LENGTH + 1, 1), np.float32), ok
    return np.stack(xs), ok


# ---------------------------------------------------------------- noise

_noise = []


def noise_pieces():
    """One-second background noise recordings (from Speech Commands)."""
    if not _noise and NOISE_DIR.exists():
        _noise.extend(load_audio(str(p)) for p in sorted(NOISE_DIR.glob("*.wav")))
    return _noise


def add_noise(y, rng, snr_db=(5.0, 25.0)):
    """Mix a random noise piece into y at a random signal-to-noise ratio."""
    pieces = noise_pieces()
    if not pieces:
        return y
    noise = pieces[rng.integers(len(pieces))]
    noise = np.resize(noise, len(y))
    snr = rng.uniform(*snr_db)
    scale = np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10) + 1e-12))
    return (y + scale * noise).astype(np.float32)


def noisy_inputs(paths, seed=42):
    """Noisy copies of recordings, for training only: the word as the model
    would hear it in a busy room."""
    rng = np.random.default_rng(seed)
    out = []
    for p in paths:
        y, _ = clean(load_audio(p))
        out.append(log_mel(normalise(add_noise(y, rng)))[..., np.newaxis])
    return np.stack(out) if out else np.zeros((0, N_MELS, CLIP_SAMPLES // HOP_LENGTH + 1, 1), np.float32)


# ---------------------------------------------------------------- datasets

def parse_fsdd_name(path):
    """`7_jackson_32.wav` -> ('7', 'jackson', 32). Other names -> (None, None, None)."""
    m = re.match(r"^(\d)_([A-Za-z0-9\-]+)_(\d+)$", Path(path).stem)
    if not m:
        return None, None, None
    return m.group(1), m.group(2).lower(), int(m.group(3))


def label_from_name(name):
    """Guess the digit from a file name or folder: '7_me_1.wav', '7/x.wav',
    'seven-2.wav'. Returns None when it can't tell."""
    p = Path(name)
    for part in [p.stem] + [q for q in reversed(p.parent.parts)]:
        part = part.lower()
        m = re.match(r"^(\d)(?:$|[_\-\s.])", part)
        if m:
            return m.group(1)
        for d, word in enumerate(["zero", "one", "two", "three", "four", "five",
                                  "six", "seven", "eight", "nine"]):
            if re.match(rf"^{word}(?:$|[_\-\s.\d])", part):
                return str(d)
    return None


def list_folder(folder):
    """Files in a <folder>/<digit>/*.wav layout as a list of dicts."""
    rows = []
    for label in LABELS:
        for wav in sorted((Path(folder) / label).glob("*.wav")):
            _, speaker, take = parse_fsdd_name(wav)
            rows.append({"path": str(wav), "label": int(label),
                         "speaker": speaker or "unknown", "take": take})
    return rows


def load_folder(folder):
    """Inputs X (N, 40, 101, 1), labels y (N,) and the file list."""
    rows = list_folder(folder)
    X, ok = prepare_many([r["path"] for r in rows])
    rows = [rows[i] for i in ok]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    return X, y, rows


# ---------------------------------------------------------------- features

def describe(source):
    """Hand-made features used for the charts (not by the model). Needs librosa."""
    import librosa
    raw = load_audio(source)
    y = trim_silence(raw)
    sr = SAMPLE_RATE
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    loud = rms >= 0.3 * rms.max() if rms.size else np.array([], bool)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=512, hop_length=HOP_LENGTH)
    n = min(len(f0), len(loud))
    voiced_f0 = f0[:n][loud[:n]] if n else f0
    return {
        "duration_s": len(y) / sr,
        "rms_db": float(20 * np.log10(np.mean(rms) + 1e-9)),
        "zcr": float(np.mean(zcr)),
        "centroid_hz": float(np.mean(centroid)),
        "pitch_hz": float(np.median(voiced_f0)) if voiced_f0.size else float("nan"),
    }
