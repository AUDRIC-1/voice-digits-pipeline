"""Prediction: one recording in, one digit out.

    from src.prediction import Predictor
    p = Predictor()
    p.predict_file("data/test/7/7_jackson_0.wav")
    # {'digit': 7, 'word': 'seven', 'confidence': 0.99, 'probabilities': [...]}

With ENGINE=tensorflow (the default when TensorFlow is installed) it runs
the Keras model; with ENGINE=lite it runs the same weights in NumPy
(src/lite.py), for servers with little memory.
"""
import os
import threading
import time

import numpy as np

from src import registry
from src.config import DIGIT_WORDS, ENGINE
from src.preprocessing import prepare


class Predictor:
    engine = ENGINE

    def __init__(self):
        self._lock = threading.Lock()
        self._stamp = None
        self.version = None
        self.loaded_at = None
        self.info = {}
        self.reload_if_changed()

    def _load(self, path):
        if ENGINE == "lite":
            from src.lite import NumpyCNN
            model = NumpyCNN(path)
            return model, model.predict
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        from src.model import INPUT_SHAPE, load
        model = load(path)
        fn = tf.function(lambda x: model(x, training=False),
                         input_signature=[tf.TensorSpec((None, *INPUT_SHAPE), tf.float32)],
                         reduce_retracing=True)
        return model, lambda X: fn(tf.convert_to_tensor(X, tf.float32)).numpy()

    def reload_if_changed(self):
        """Load the live model if a new one was registered since last time."""
        stamp = registry.stamp()
        if stamp == self._stamp:
            return False
        with self._lock:
            if stamp == self._stamp:
                return False
            version, path, info = registry.current()
            model, fn = self._load(path)
            fn(np.zeros((1, 40, 101, 1), np.float32))  # warm up
            self.model, self._fn = model, fn
            self.version, self.info, self._stamp = version, info, stamp
            self.loaded_at = time.time()
            return True

    def predict_inputs(self, X):
        """Probabilities for prepared inputs of shape (N, 40, 101, 1)."""
        self.reload_if_changed()
        if len(X) == 0:
            return np.zeros((0, len(DIGIT_WORDS)), np.float32)
        return np.asarray(self._fn(np.asarray(X, np.float32)))

    def predict_file(self, source):
        """source: a path, raw bytes or a file object."""
        proba = self.predict_inputs(prepare(source)[np.newaxis])[0]
        digit = int(proba.argmax())
        return {
            "digit": digit,
            "word": DIGIT_WORDS[digit],
            "confidence": float(proba[digit]),
            "probabilities": [round(float(p), 4) for p in proba],
            "model_version": self.version,
        }


def predict_file(path):
    """One-off helper for scripts and the notebook."""
    return Predictor().predict_file(path)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(f, predict_file(f))
