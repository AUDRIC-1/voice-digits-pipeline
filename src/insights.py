"""Features for the charts in the notebook and the app.

Builds results/features.csv (one row per recording) and
results/insights.json (the summaries the web app draws).

Run:  python -m src.insights
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (DATA_DIR, DIGIT_WORDS, FEATURES_CSV, NEW_VOICE_DIR,
                        RESULTS_DIR, TEST_DIR, TRAIN_DIR)
from src.preprocessing import clean, describe, list_folder, load_audio, log_mel

INSIGHTS_JSON = RESULTS_DIR / "insights.json"
FEATURES = ["duration_s", "zcr", "centroid_hz", "pitch_hz", "rms_db"]


SC_GROUP = "speech commands"


def voice_group(speaker):
    """FSDD's six speakers by name; Speech Commands' thousands as one group."""
    return SC_GROUP if str(speaker).startswith("sc-") else speaker


def build_features():
    rows = []
    for split, folder in (("train", TRAIN_DIR), ("test", TEST_DIR), ("new_voice", NEW_VOICE_DIR)):
        for r in list_folder(folder):
            rows.append({"split": split, "file": Path(r["path"]).relative_to(DATA_DIR).as_posix(),
                         "label": r["label"], "speaker": r["speaker"], **describe(r["path"])})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    return df


def _quartiles(s):
    s = s.dropna()
    q = np.percentile(s, [5, 25, 50, 75, 95]) if len(s) else [None] * 5
    return {"p5": q[0], "q1": q[1], "median": q[2], "q3": q[3], "p95": q[4], "mean": float(s.mean())}


def build_insights(df=None):
    df = df if df is not None else pd.read_csv(FEATURES_CSV)
    df = df.assign(speaker=df.speaker.map(voice_group))
    out = {"digits": DIGIT_WORDS, "features": FEATURES}

    out["counts"] = {split: [int((g.label == d).sum()) for d in range(10)]
                     for split, g in df.groupby("split")}
    out["speakers"] = {sp: {split: int(n) for split, n in g.split.value_counts().items()}
                       for sp, g in df.groupby("speaker")}

    out["by_digit"] = {f: [_quartiles(df[df.label == d][f]) for d in range(10)] for f in FEATURES}
    out["by_speaker"] = {f: {sp: _quartiles(g[f]) for sp, g in df.groupby("speaker")} for f in FEATURES}

    # Average spectrogram of each digit (training set), every second frame
    train = df[df.split == "train"]
    means = []
    for d in range(10):
        files = train[train.label == d].file.head(60)
        specs = [log_mel(clean(load_audio(str(DATA_DIR / f)))[0]) for f in files]
        means.append(np.round(np.mean(specs, axis=0)[:, ::2], 3).tolist())
    out["mean_spectrogram"] = means

    with open(INSIGHTS_JSON, "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    data = build_features()
    print(data.groupby("split").size())
    build_insights(data)
    print(f"wrote {FEATURES_CSV} and {INSIGHTS_JSON}")
