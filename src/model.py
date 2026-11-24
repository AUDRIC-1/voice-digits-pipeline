"""Step 3 and 4: model creation, training, testing and retraining.

The model is a small convolutional network (CNN) that reads a log-mel
spectrogram like a one-channel image and outputs one probability per digit.

Ways the training is kept from overfitting:
- data augmentation (random time shift plus SpecAugment-style frequency and
  time masks), done fresh on every batch
- batch normalisation (momentum 0.9), dropout and L2 weight decay
- Adam optimiser, learning rate halved when validation loss stalls
- early stopping that keeps the best epoch's weights

Retraining does not start from zero. It loads the current model and keeps
training it (fine-tuning) with a lower learning rate on the new recordings
mixed with the original training data, so it learns the new voices without
forgetting the old ones.
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import absl.logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src import metrics
from src.config import LABELS, N_MELS, SEED

absl.logging.set_verbosity(absl.logging.ERROR)   # hides the "HDF5 is legacy" notice
tf.get_logger().setLevel("ERROR")                # and retracing notices
keras = tf.keras
TIME_STEPS = 101
INPUT_SHAPE = (N_MELS, TIME_STEPS, 1)


def set_seed(seed=SEED):
    keras.utils.set_random_seed(seed)


def build_model(l2=1e-4, dropout=0.3):
    reg = keras.regularizers.l2(l2)

    def block(x, filters):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                                kernel_regularizer=reg)(x)
        # momentum 0.9 instead of Keras' 0.99: with only ~30 batches per epoch
        # the default running averages lag, and validation sits at chance
        x = keras.layers.BatchNormalization(momentum=0.9)(x)
        return keras.layers.Activation("relu")(x)

    inputs = keras.Input(shape=INPUT_SHAPE, name="log_mel")
    x = block(inputs, 16)
    x = keras.layers.MaxPooling2D(2)(x)
    x = block(x, 32)
    x = keras.layers.MaxPooling2D(2)(x)
    x = block(x, 64)
    x = keras.layers.MaxPooling2D(2)(x)
    x = block(x, 64)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax", name="digit")(x)
    return keras.Model(inputs, outputs, name="digit_cnn")


# ---------------------------------------------------------------- data

def augment(x, y):
    """Random time shift and one frequency mask and one time mask."""
    shift = tf.random.uniform([], -8, 9, dtype=tf.int32)
    x = tf.roll(x, shift, axis=1)

    f = tf.random.uniform([], 0, 7, dtype=tf.int32)
    f0 = tf.random.uniform([], 0, N_MELS - 6, dtype=tf.int32)
    rows = tf.range(N_MELS)[:, None, None]
    x = x * tf.cast((rows < f0) | (rows >= f0 + f), x.dtype)

    t = tf.random.uniform([], 0, 11, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, TIME_STEPS - 10, dtype=tf.int32)
    cols = tf.range(TIME_STEPS)[None, :, None]
    x = x * tf.cast((cols < t0) | (cols >= t0 + t), x.dtype)
    return x, y


def make_dataset(X, y, batch_size=64, training=False, sample_weight=None):
    parts = (X, y) if sample_weight is None else (X, y, sample_weight)
    ds = tf.data.Dataset.from_tensor_slices(parts)
    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
        if sample_weight is None:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda a, b, w: (*augment(a, b), w),
                        num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_train_val(X, y, groups=None, val_size=0.15, seed=SEED):
    """Stratified split. `groups` (e.g. speaker) is added to the strata so
    every speaker shows up in both parts."""
    strata = y if groups is None else np.array([f"{a}_{b}" for a, b in zip(y, groups)])
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=val_size, stratify=strata, random_state=seed)
    return tr, va


# ---------------------------------------------------------------- training

class ProgressCallback(keras.callbacks.Callback):
    """Reports each finished epoch to a function (used by the web app)."""

    def __init__(self, report):
        super().__init__()
        self.report = report

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        self.report(epoch + 1, logs)


def _callbacks(patience, on_epoch=None):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, patience // 2), min_lr=1e-5),
    ]
    if on_epoch:
        cbs.append(ProgressCallback(on_epoch))
    return cbs


def train(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=1e-3,
          patience=8, on_epoch=None, verbose=2):
    """Train a new model from random weights."""
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(make_dataset(X_train, y_train, batch_size, training=True),
                        validation_data=make_dataset(X_val, y_val, batch_size),
                        epochs=epochs, callbacks=_callbacks(patience, on_epoch),
                        shuffle=False, verbose=verbose)   # the dataset already shuffles
    return model, _clean_history(history.history)


def fine_tune(model, X_train, y_train, X_val, y_val, sample_weight=None,
              epochs=15, batch_size=64, lr=3e-4, patience=4, on_epoch=None, verbose=2):
    """Keep training an existing model (retraining). The new recordings can
    be given a larger sample_weight so a few of them still count."""
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(make_dataset(X_train, y_train, batch_size, training=True,
                                     sample_weight=sample_weight),
                        validation_data=make_dataset(X_val, y_val, batch_size),
                        epochs=epochs, callbacks=_callbacks(patience, on_epoch),
                        shuffle=False, verbose=verbose)   # the dataset already shuffles
    return model, _clean_history(history.history)


def _clean_history(h):
    return {k: [float(v) for v in vals] for k, vals in h.items()}


# ---------------------------------------------------------------- testing

def predict_proba(model, X, batch_size=256):
    if len(X) == 0:
        return np.zeros((0, len(LABELS)), np.float32)
    return model.predict(X, batch_size=batch_size, verbose=0)


def evaluate(model, X, y):
    """All the metrics used in the notebook and in the app."""
    return metrics.from_proba(predict_proba(model, X), y)


def accuracy(model, X, y):
    if len(y) == 0:
        return None
    return float((predict_proba(model, X).argmax(axis=1) == y).mean())


# ---------------------------------------------------------------- files

def save(model, path, info=None):
    path = str(path)
    model.save(path)
    if info is not None:
        with open(os.path.splitext(path)[0] + "_info.json", "w") as f:
            json.dump(info, f, indent=2)


def load(path):
    return keras.models.load_model(str(path), compile=False)
