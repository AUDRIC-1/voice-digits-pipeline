"""Audio maths in plain NumPy/SciPy, so the app needs neither librosa nor numba.

These functions copy what librosa 1.0 does with the settings this project
uses (checked in scripts/check_lite.py), so a model trained on librosa
spectrograms gets the same inputs here.
"""
import io
from functools import lru_cache
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AMIN = 1e-10


def read(source, sr):
    """Read a path, bytes or file object as mono float32 at sample rate sr."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, native = sf.read(source, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if native != sr:
        g = gcd(int(native), int(sr))
        y = resample_poly(y, sr // g, int(native) // g).astype(np.float32)
    return y


def _frames(y, frame_length, hop_length):
    """Centred frames, padded with zeros like librosa (center=True, 'constant')."""
    y = np.pad(y, (frame_length // 2, frame_length // 2))
    n = 1 + (len(y) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n)[:, None]
    return y[idx]


def power_to_db(S, ref, top_db=80.0):
    db = 10.0 * np.log10(np.maximum(AMIN, S)) - 10.0 * np.log10(max(AMIN, ref))
    if top_db is not None:
        db = np.maximum(db, db.max() - top_db)
    return db


def trim(y, top_db, frame_length=2048, hop_length=512):
    """Cut leading and trailing silence (librosa.effects.trim)."""
    if len(y) == 0:
        return y
    mse = np.mean(_frames(y.astype(np.float32), frame_length, hop_length) ** 2, axis=1)
    db = power_to_db(mse, ref=mse.max(), top_db=None)
    loud = np.flatnonzero(db > -top_db)
    if loud.size == 0:
        return y[:0]
    start = loud[0] * hop_length
    end = min(len(y), (loud[-1] + 1) * hop_length)
    return y[start:end]


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    mel = f / f_sp
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


@lru_cache(maxsize=4)
def mel_basis(sr, n_fft, n_mels, fmax):
    """Slaney-style mel filterbank (librosa.filters.mel defaults)."""
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def melspectrogram(y, sr, n_fft, hop_length, n_mels, fmax):
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)   # periodic Hann
    spec = np.abs(np.fft.rfft(_frames(y, n_fft, hop_length) * window, axis=1)) ** 2
    return mel_basis(sr, n_fft, n_mels, fmax) @ spec.T.astype(np.float32)
