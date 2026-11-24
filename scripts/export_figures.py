"""Save the notebook's charts as PNG files in results/figures/ (used by the README).

    python scripts/export_figures.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "voice_digits_pipeline.ipynb"
OUT = ROOT / "results" / "figures"

# A line of code that appears in the cell -> file name
NAMES = {
    "1. raw:": "preprocessing_steps.png",
    'y="duration_s"': "feature_duration.png",
    'y="zcr"': "feature_zcr_centroid.png",
    'y="pitch_hz"': "feature_pitch_loudness.png",
    "Average log-mel spectrogram": "mean_spectrograms.png",
    'history["val_loss"]': "learning_curves.png",
    "Test set, known voices": "confusion_matrices.png",
    "Before: george held-aside": "retraining_confusion.png",
}


def main():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        name = next((n for key, n in NAMES.items() if key in src), None)
        if not name:
            continue
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png:
                (OUT / name).write_bytes(base64.b64decode(png))
                print("saved", OUT / name)
                break


if __name__ == "__main__":
    main()
