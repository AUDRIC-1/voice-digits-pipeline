"""Export the trained Keras model for the light engine (run once, needs TensorFlow).

    python scripts/export_lite.py

Writes models/lite/digit_cnn.npz (weights, batch norm folded in),
models/lite/digit_cnn_info.json and models/lite/embeddings.npz (the 64-number
embeddings of the training, test and new-voice clips, used for last-layer
retraining and evaluation on a small server), and checks that NumPy and Keras
give the same probabilities on the test set.

It runs in two processes to save memory: only the first loads TensorFlow.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, lite  # noqa: E402

KERAS_PROBA = lite.LITE_DIR / "keras_test_proba.npy"


def keras_part():
    """Weights and Keras' answers on the test set."""
    from src import model as M
    from src.preprocessing import load_folder
    keras_model = M.load(config.BASE_MODEL_PATH)
    lite.LITE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(lite.BASE_LITE_PATH, **lite.export_from_keras(keras_model))
    X_te, _, _ = load_folder(config.TEST_DIR)
    np.save(KERAS_PROBA, M.predict_proba(keras_model, X_te))
    print("keras part done")


def numpy_part():
    """Compare with Keras, then store the embeddings. No TensorFlow here."""
    from src.preprocessing import load_folder, noisy_inputs
    cnn = lite.NumpyCNN(lite.BASE_LITE_PATH)
    X_te, y_te, rows_te = load_folder(config.TEST_DIR)
    p_keras, p_numpy = np.load(KERAS_PROBA), cnn.predict(X_te)
    diff = float(np.abs(p_keras - p_numpy).max())
    same = float((p_keras.argmax(1) == p_numpy.argmax(1)).mean())
    print(f"test set: largest probability difference {diff:.2e}, same answer on {same:.1%} of clips")
    print(f"accuracy keras {(p_keras.argmax(1) == y_te).mean():.4f}, numpy {(p_numpy.argmax(1) == y_te).mean():.4f}")
    assert diff < 1e-3 and same == 1.0, "NumPy model does not match Keras"
    test_E = cnn.embed(X_te).astype(np.float32)
    del X_te

    X_tr, y_tr, rows_tr = load_folder(config.TRAIN_DIR)
    train_E = cnn.embed(X_tr).astype(np.float32)
    del X_tr
    rng = np.random.default_rng(config.SEED)
    noisy = np.array([i for i in range(len(rows_tr)) if rng.random() < 0.5])
    noisy_E = np.concatenate([cnn.embed(noisy_inputs([rows_tr[i]["path"] for i in noisy[k:k + 500]]))
                              for k in range(0, len(noisy), 500)]).astype(np.float32)
    X_new, y_new, _ = load_folder(config.NEW_VOICE_DIR)

    np.savez_compressed(
        lite.EMBEDDINGS_PATH,
        train_E=np.concatenate([train_E, noisy_E]),
        train_y=np.concatenate([y_tr, y_tr[noisy]]),
        train_noisy=np.concatenate([np.zeros(len(train_E), bool), np.ones(len(noisy), bool)]),
        test_E=test_E, test_y=y_te,
        test_unseen=np.array([r["speaker"].startswith("sc-") for r in rows_te]),
        new_E=cnn.embed(X_new).astype(np.float32), new_y=y_new,
    )
    info = json.loads(config.BASE_MODEL_INFO.read_text())
    info.update({"engine": "lite",
                 "note": "same weights as models/digit_cnn.h5, run in NumPy (scripts/export_lite.py)",
                 "lite_check": {"max_probability_difference": diff, "same_answer": same}})
    lite.BASE_LITE_INFO.write_text(json.dumps(info, indent=2))
    KERAS_PROBA.unlink()
    for p in (lite.BASE_LITE_PATH, lite.EMBEDDINGS_PATH):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step == "keras":
        keras_part()
    elif step == "numpy":
        numpy_part()
    else:
        for part in ("keras", "numpy"):
            subprocess.run([sys.executable, __file__, part], check=True)
