"""Paths and settings shared by every part of the pipeline.

Anything that changes at run time (uploads, the database, retrained models)
can be moved with the STORAGE_DIR environment variable, which is how the
Docker containers share one volume.
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data that ships with the repo
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
NEW_VOICE_DIR = DATA_DIR / "new_voice"   # a speaker the first model never hears
NOISE_DIR = DATA_DIR / "noise"           # background noise mixed into training clips

# The first model, trained in the notebook. It is never overwritten.
MODELS_DIR = ROOT / "models"
BASE_MODEL_PATH = MODELS_DIR / "digit_cnn.h5"
BASE_MODEL_INFO = MODELS_DIR / "digit_cnn_info.json"

# Things the running app writes
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", ROOT / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
VERSIONS_DIR = STORAGE_DIR / "models"
REGISTRY_PATH = VERSIONS_DIR / "registry.json"
DB_PATH = STORAGE_DIR / "app.db"

RESULTS_DIR = ROOT / "results"
FEATURES_CSV = RESULTS_DIR / "features.csv"

# Audio settings
SAMPLE_RATE = 8000          # the dataset is recorded at 8 kHz
CLIP_SECONDS = 1.0          # every clip is padded or cut to one second
N_MELS = 40
N_FFT = 256                 # 32 ms window
HOP_LENGTH = 80             # 10 ms step
TOP_DB = 30                 # silence threshold for trimming
LABELS = [str(d) for d in range(10)]
DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]

# Retraining trigger
RETRAIN_MIN_NEW_SAMPLES = int(os.environ.get("RETRAIN_MIN_NEW_SAMPLES", "100"))
RETRAIN_ACCURACY_TRIGGER = float(os.environ.get("RETRAIN_ACCURACY_TRIGGER", "0.85"))
RETRAIN_MIN_FOR_ACCURACY = int(os.environ.get("RETRAIN_MIN_FOR_ACCURACY", "20"))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "0") == "1"
# A retrained model replaces the current one only if it does not lose more
# than this much accuracy on the fixed test set.
MAX_TEST_DROP = float(os.environ.get("MAX_TEST_DROP", "0.02"))

SEED = 42


def ensure_dirs():
    for d in (UPLOAD_DIR, VERSIONS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# "tensorflow" (full training) or "lite" (NumPy model, last-layer retraining,
# fits in 512 MB). Default: tensorflow if it is installed.
ENGINE = os.environ.get("ENGINE") or ("tensorflow" if importlib.util.find_spec("tensorflow") else "lite")
