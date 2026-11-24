"""Step 1: data acquisition.

Downloads the Free Spoken Digit Dataset (FSDD) and sorts it into
data/train, data/test and data/new_voice, one folder per digit.

FSDD file names look like `7_jackson_32.wav`: digit, speaker, take number.
- Takes 0 to 4 of every digit go to the test set (the split the dataset's
  authors recommend), takes 5 to 49 go to training.
- One speaker is kept out completely and goes to data/new_voice. The first
  model never hears this voice, so it stands in for new users who arrive
  after the app is live. It is the data used to show retraining.

Then the digit words from Google Speech Commands are added to train and
test (see src/speech_commands.py): thousands of other voices, recorded in
browsers like the app's Record button.

Run:  python -m src.data_acquisition
"""
import argparse
import io
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from src.config import DATA_DIR, NEW_VOICE_DIR, TEST_DIR, TRAIN_DIR

FSDD_VERSION = "v1.0.10"
FSDD_URL = ("https://github.com/Jakobovski/free-spoken-digit-dataset/"
            f"archive/refs/tags/{FSDD_VERSION}.zip")
RAW_DIR = DATA_DIR / "raw"
NEW_SPEAKER = "george"
TEST_TAKES = range(0, 5)


def download(raw_dir=RAW_DIR, url=FSDD_URL):
    """Download the dataset zip once and keep only the .wav files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = list(raw_dir.glob("*.wav"))
    if len(existing) >= 3000:
        print(f"Already downloaded: {len(existing)} files in {raw_dir}")
        return raw_dir
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in zf.namelist():
            if "/recordings/" in name and name.endswith(".wav"):
                (raw_dir / Path(name).name).write_bytes(zf.read(name))
    print(f"Saved {len(list(raw_dir.glob('*.wav')))} recordings to {raw_dir}")
    return raw_dir


def parse_name(path):
    digit, speaker, take = Path(path).stem.split("_")
    return digit, speaker, int(take)


def split(raw_dir=RAW_DIR, new_speaker=NEW_SPEAKER):
    """Copy the raw files into train / test / new_voice folders."""
    for d in (TRAIN_DIR, TEST_DIR, NEW_VOICE_DIR):
        if d.exists():
            shutil.rmtree(d)
    counts = Counter()
    for wav in sorted(raw_dir.glob("*.wav")):
        digit, speaker, take = parse_name(wav)
        if speaker == new_speaker:
            target = NEW_VOICE_DIR
        elif take in TEST_TAKES:
            target = TEST_DIR
        else:
            target = TRAIN_DIR
        (target / digit).mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, target / digit / wav.name)
        counts[target.name] += 1
    for name, n in counts.items():
        print(f"{name:10s} {n:5d} files")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-speaker", default=NEW_SPEAKER)
    parser.add_argument("--fsdd-only", action="store_true",
                        help="skip the Speech Commands download (2.3 GB)")
    args = parser.parse_args()
    split(download(), args.new_speaker)
    if not args.fsdd_only:
        from src import speech_commands
        speech_commands.build(speech_commands.download())


if __name__ == "__main__":
    main()
