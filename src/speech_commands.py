"""Second data source: Google Speech Commands v0.02 (CC BY 4.0).

About 2,600 people recorded single words through a web browser on their
own computers and phones. Only the ten digit words are used. That is the
same kind of recording the app's Record button makes, from far more voices
and accents than FSDD's six speakers.

Run:  python -m src.speech_commands     (downloads ~2.3 GB once, keeps ~1 GB)

The dataset's own testing list keeps test speakers apart from training
speakers, so the test clips here are voices the model has never heard.
"""
import argparse
import hashlib
import random
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf

from src.config import DATA_DIR, DIGIT_WORDS, NOISE_DIR, SAMPLE_RATE, TEST_DIR, TOP_DB, TRAIN_DIR
from src.preprocessing import load_audio

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
RAW = DATA_DIR / "raw" / "speech_commands"
WORDS = set(DIGIT_WORDS)
TRAIN_PER_DIGIT = 400
TEST_PER_DIGIT = 100


def download(raw=RAW, url=URL):
    """Stream the archive and keep only the digit words, the speaker
    lists and the background noise."""
    if (raw / "testing_list.txt").exists() and len(list(raw.glob("*/*.wav"))) > 30000:
        print("already downloaded:", raw)
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    n = 0
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            name = m.name.lstrip("./")
            top = name.split("/")[0]
            if not m.isfile():
                continue
            if top in WORDS or top == "_background_noise_" or name in (
                    "testing_list.txt", "validation_list.txt", "LICENSE", "README.md"):
                target = raw / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src:
                    target.write_bytes(src.read())
                n += 1
                if n % 5000 == 0:
                    print(f"  {n} files")
    print(f"kept {n} files in {raw}")
    return raw


def _write(y, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, SAMPLE_RATE, subtype="PCM_16")


def _clip(path):
    """Resample to 8 kHz and cut the silence, keeping 0.1 s on each side."""
    y = load_audio(str(path))
    _, (a, b) = librosa.effects.trim(y, top_db=TOP_DB)
    if b - a < 0.15 * SAMPLE_RATE:
        return None
    return y[max(0, a - SAMPLE_RATE // 10):min(len(y), b + SAMPLE_RATE // 10)]


def build(raw=RAW, train_n=TRAIN_PER_DIGIT, test_n=TEST_PER_DIGIT, seed=42):
    """Add a sample of the digit words to data/train and data/test.
    File names follow the FSDD pattern: 7_sc-<speaker>_<take>.wav"""
    test_list = set((raw / "testing_list.txt").read_text().split())
    rng = random.Random(seed)
    counts = defaultdict(int)
    for d, word in enumerate(DIGIT_WORDS):
        files = sorted((raw / word).glob("*.wav"))
        test = [f for f in files if f"{word}/{f.name}" in test_list]
        train = [f for f in files if f"{word}/{f.name}" not in test_list]
        for split, pool, n, out in (("train", train, train_n, TRAIN_DIR), ("test", test, test_n, TEST_DIR)):
            # one clip per speaker first, so the sample has as many voices as possible
            by_speaker = defaultdict(list)
            for f in pool:
                by_speaker[f.name.split("_")[0]].append(f)
            speakers = sorted(by_speaker)
            rng.shuffle(speakers)
            picked = [rng.choice(by_speaker[s]) for s in speakers][: n * 2]
            kept = 0
            for f in picked:
                if kept >= n:
                    break
                y = _clip(f)
                if y is None:
                    continue
                speaker, _, take = f.stem.split("_")
                _write(y, out / str(d) / f"{d}_sc-{speaker}_{take}.wav")
                kept += 1
            counts[(split, d)] = kept
    for split in ("train", "test"):
        print(split, [counts[(split, d)] for d in range(10)])

    # background noise, cut into 1-second pieces for training-time mixing
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    pieces = 0
    for f in sorted((raw / "_background_noise_").glob("*.wav")):
        y = load_audio(str(f))
        for i in range(0, len(y) - SAMPLE_RATE, SAMPLE_RATE * 3)[:20]:
            _write(y[i:i + SAMPLE_RATE], NOISE_DIR / f"{f.stem}_{i // SAMPLE_RATE}.wav")
            pieces += 1
    print(f"{pieces} noise pieces in {NOISE_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=TRAIN_PER_DIGIT)
    ap.add_argument("--test", type=int, default=TEST_PER_DIGIT)
    args = ap.parse_args()
    build(download(), args.train, args.test)


if __name__ == "__main__":
    main()
