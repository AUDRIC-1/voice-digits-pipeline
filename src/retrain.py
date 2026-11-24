"""Step 5: uploading new data, the retraining trigger and the retraining job.

Upload: each file is checked (readable, has speech, one word long), its
digit is taken from the form or the file name, and it is saved under
storage/uploads/<digit>/ with a row in the database. About one file in five
is kept aside ("holdout") to check the retrained model on voices it did not
train on.

Trigger: retraining is needed when either
  - at least RETRAIN_MIN_NEW_SAMPLES new recordings are waiting, or
  - the live model gets fewer than RETRAIN_ACCURACY_TRIGGER of the waiting
    recordings right (checked once at least RETRAIN_MIN_FOR_ACCURACY are in).
The app shows the reason; with AUTO_RETRAIN=1 it also starts the job itself.

Job: load the live model and fine-tune it on the original training data plus
every uploaded recording, then test it. The new model goes live only if it
is not worse on the fixed test set (within MAX_TEST_DROP) and not worse on
the held-aside uploads. Otherwise it is saved but the old one stays live.
"""
import hashlib
import io
import re
import threading
import traceback
import zipfile
from pathlib import Path

import numpy as np

from src import database, metrics, registry
from src.config import (AUTO_RETRAIN, ENGINE, MAX_TEST_DROP, NEW_VOICE_DIR,
                        RETRAIN_ACCURACY_TRIGGER, RETRAIN_MIN_FOR_ACCURACY,
                        RETRAIN_MIN_NEW_SAMPLES, SEED, TEST_DIR, TRAIN_DIR, UPLOAD_DIR)
from src.preprocessing import (AUDIO_EXTENSIONS, AudioError, clean, label_from_name,
                               load_audio, load_folder, log_mel, noisy_inputs,
                               parse_fsdd_name, prepare_many)

_cache = {}
_cache_lock = threading.Lock()


def _group(speaker):
    return "sc" if speaker.startswith("sc-") else speaker


def dataset(name):
    """Prepared inputs for a folder in the repo, kept in memory after the first use."""
    with _cache_lock:
        if name not in _cache:
            folder = {"train": TRAIN_DIR, "test": TEST_DIR, "new_voice": NEW_VOICE_DIR}[name]
            _cache[name] = load_folder(folder)
        return _cache[name]


def dataset_proba(name, predictor):
    """Live-model probabilities and labels for 'test', 'unseen' or 'new_voice'.
    The light engine uses the stored embeddings, so a slow server doesn't
    reprocess 1,250 recordings for every evaluation."""
    if ENGINE == "lite":
        from src import lite
        z = lite.embeddings()
        if name == "new_voice":
            E, y = z["new_E"], z["new_y"]
        else:
            keep = z["test_unseen"] if name == "unseen" else np.ones(len(z["test_y"]), bool)
            E, y = z["test_E"][keep], z["test_y"][keep]
        predictor.reload_if_changed()
        return predictor.model.head_proba(E), y
    X, y, rows = dataset("new_voice" if name == "new_voice" else "test")
    if name == "unseen":
        keep = [i for i, r in enumerate(rows) if r["speaker"].startswith("sc-")]
        X, y = X[keep], y[keep]
    return predictor.predict_inputs(X), y


# ---------------------------------------------------------------- uploads

def _expand(files):
    """Unpack zips. Yields (name, bytes)."""
    for name, data in files:
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for inner in zf.namelist():
                        if Path(inner).suffix.lower() in AUDIO_EXTENSIONS and not inner.startswith("__MACOSX"):
                            yield inner, zf.read(inner)
            except zipfile.BadZipFile:
                yield name, b""
        else:
            yield name, data


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(name).name)[-80:]


def save_batch(files, label=None, speaker=None, predictor=None):
    """Save a bulk upload. `files` is a list of (filename, bytes)."""
    batch_id = database.execute(
        "INSERT INTO batches (uploaded_at, n_saved, n_rejected, n_duplicate) VALUES (?,0,0,0)",
        (registry.now(),))
    saved, rejected, duplicate = [], [], 0
    inputs = []
    for name, data in _expand(files):
        digit = str(label) if label not in (None, "", "auto") else label_from_name(name)
        if digit is None:
            rejected.append({"file": name, "reason": "no digit given and none in the file name"})
            continue
        try:
            y, spoken = clean(load_audio(data))
        except AudioError as exc:
            rejected.append({"file": name, "reason": str(exc)})
            continue
        sha = hashlib.sha256(data).hexdigest()
        if database.one("SELECT id FROM uploads WHERE sha256=?", (sha,)):
            duplicate += 1
            continue
        who = speaker or parse_fsdd_name(name)[1] or "unknown"
        target = UPLOAD_DIR / digit / f"b{batch_id}_{_safe(name)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        holdout = int(int(sha[:8], 16) % 5 == 0)
        database.execute(
            "INSERT INTO uploads (batch_id, filename, stored_path, label, speaker, sha256, "
            "spoken_s, uploaded_at, holdout) VALUES (?,?,?,?,?,?,?,?,?)",
            (batch_id, name, str(target), int(digit), who, sha, spoken, registry.now(), holdout))
        saved.append({"file": name, "label": int(digit), "holdout": bool(holdout)})
        inputs.append(log_mel(y)[..., np.newaxis])

    acc = None
    if predictor is not None and inputs:
        proba = predictor.predict_inputs(np.stack(inputs))
        acc = float((proba.argmax(1) == np.array([s["label"] for s in saved])).mean())
    trig = check_trigger(predictor)
    database.execute(
        "UPDATE batches SET n_saved=?, n_rejected=?, n_duplicate=?, model_version=?, "
        "accuracy_before=?, trigger_fired=?, trigger_reason=? WHERE id=?",
        (len(saved), len(rejected), duplicate, predictor.version if predictor else None,
         acc, int(trig["fired"]), "; ".join(trig["reasons"]), batch_id))

    started = None
    if trig["fired"] and AUTO_RETRAIN:
        started = start("retrain", "automatic: " + "; ".join(trig["reasons"]), predictor)
    return {"batch_id": batch_id, "saved": len(saved), "rejected": rejected,
            "duplicates": duplicate, "batch_accuracy": acc, "files": saved[:200],
            "trigger": trig, "auto_retrain_run": started}


def _load_uploads(rows):
    X, ok = prepare_many([r["stored_path"] for r in rows])
    rows = [rows[i] for i in ok]
    return X, np.array([r["label"] for r in rows], dtype=np.int64), rows


def check_trigger(predictor=None):
    pending = database.pending_uploads()
    n = len(pending)
    reasons, acc = [], None
    if n >= RETRAIN_MIN_NEW_SAMPLES:
        reasons.append(f"{n} new recordings are waiting (limit {RETRAIN_MIN_NEW_SAMPLES})")
    if predictor is not None and n >= RETRAIN_MIN_FOR_ACCURACY:
        X, y, _ = _load_uploads(pending)
        if len(y):
            acc = float((predictor.predict_inputs(X).argmax(1) == y).mean())
            if acc < RETRAIN_ACCURACY_TRIGGER:
                reasons.append(f"the live model gets only {acc:.0%} of the new recordings right "
                               f"(below {RETRAIN_ACCURACY_TRIGGER:.0%})")
    return {"fired": bool(reasons), "reasons": reasons, "pending": n,
            "accuracy_on_pending": acc, "auto_retrain": AUTO_RETRAIN,
            "rules": {"min_new_samples": RETRAIN_MIN_NEW_SAMPLES,
                      "accuracy_below": RETRAIN_ACCURACY_TRIGGER,
                      "min_for_accuracy": RETRAIN_MIN_FOR_ACCURACY}}


# ---------------------------------------------------------------- the job

def start(mode, reason, predictor=None):
    """Start a training run in the background. Returns the run id, or None
    if one is already running."""
    version = predictor.version if predictor else registry.current()[0]
    run_id = database.start_run(mode, reason, version)
    if run_id is None:
        return None
    threading.Thread(target=_job, args=(run_id, mode), daemon=True,
                     name=f"train-{run_id}").start()
    return run_id


def _job(run_id, mode):
    if ENGINE == "lite":
        return _job_lite(run_id, mode)
    return _job_tf(run_id, mode)


def _uploads_for_job(mode):
    pending_ids = {r["id"] for r in database.pending_uploads()}
    uploads = database.rows("SELECT * FROM uploads ORDER BY id")
    learn = [r for r in uploads if not r["holdout"]]
    check = [r for r in uploads if r["holdout"]]
    n_new = sum(1 for r in learn if r["id"] in pending_ids)
    if mode == "retrain" and n_new == 0:
        raise RuntimeError("no new recordings to learn from; upload some first")
    return pending_ids, learn, check, n_new


def _finish(run_id, mode, pending_ids, n_new, n_replay, before, after, test_metrics, history, info, save):
    """Promotion rule, new version, database rows. Shared by both engines."""
    (test_before, new_before), (test_after, new_after) = before, after
    ok_test = test_after >= test_before - MAX_TEST_DROP
    ok_new = new_before is None or new_after >= new_before
    promote = ok_test and ok_new

    version = registry.next_version()
    path = registry.path_for(version)
    info.update({"version": version, "created_at": registry.now(), "mode": mode, "run_id": run_id,
                 "test_metrics": test_metrics, "uploads_holdout_accuracy": new_after, "history": history})
    save(path, info)
    note = f"{mode} run {run_id}" + ("" if promote else " (not used: not better)")
    registry.add(version, path, note, make_current=promote)

    if pending_ids:
        marks = ",".join(str(i) for i in pending_ids)
        database.execute(f"UPDATE uploads SET used_in_run=? WHERE id IN ({marks})", (run_id,))

    why = []
    if not ok_test:
        why.append(f"test accuracy fell from {test_before:.1%} to {test_after:.1%}")
    if not ok_new:
        why.append(f"held-aside uploads fell from {new_before:.1%} to {new_after:.1%}")
    database.update_run(
        run_id, status="done" if promote else "kept_old", finished_at=registry.now(),
        new_version=version, promoted=int(promote), n_new=n_new, n_replay=n_replay,
        epochs=len(history.get("loss", [])), test_acc_before=test_before,
        test_acc_after=test_after, new_acc_before=new_before, new_acc_after=new_after,
        metrics={"test": test_metrics, "history": history},
        message=(f"version {version} is live" if promote
                 else "kept the old model: " + "; ".join(why)),
        progress={"stage": "finished"})


def _job_lite(run_id, mode):
    """Light engine: keep the convolution layers, retrain the last layer.
    'retrain' starts from the live model's last layer, 'train' from zero."""
    import json

    from src import lite

    def progress(stage, **extra):
        database.update_run(run_id, progress={"stage": stage, **extra})

    try:
        progress("loading the stored training embeddings")
        z = lite.embeddings()
        E_tr, y_tr, noisy_tr = z["train_E"], z["train_y"], z["train_noisy"]
        E_te, y_te = z["test_E"], z["test_y"]
        pending_ids, learn, check, n_new = _uploads_for_job(mode)

        base_version, base_path, _ = registry.current()
        base = lite.NumpyCNN(base_path)
        progress("preparing the uploaded recordings")
        X_up, y_up, learn = _load_uploads(learn) if learn else (np.zeros((0, 40, 101, 1), np.float32), y_tr[:0], [])
        X_ck, y_ck, check = _load_uploads(check) if check else (np.zeros((0, 40, 101, 1), np.float32), y_tr[:0], [])
        E_up, E_ck = base.embed(X_up), base.embed(X_ck)
        test_before = metrics.accuracy(base.head_proba(E_te), y_te)
        new_before = metrics.accuracy(base.head_proba(E_ck), y_ck) if len(y_ck) else None

        # 15% of the clean original clips and of the uploads are kept for validation
        rng = np.random.default_rng(SEED)
        E = np.concatenate([E_tr, E_up])
        y = np.concatenate([y_tr, y_up])
        is_up = np.concatenate([np.zeros(len(E_tr), bool), np.ones(len(E_up), bool)])
        clean_idx = np.flatnonzero(~np.concatenate([noisy_tr, np.zeros(len(E_up), bool)]))
        va = rng.choice(clean_idx, size=max(1, int(0.15 * len(clean_idx))), replace=False)
        tr = np.setdiff1d(np.arange(len(E)), va)
        n_clean = int((~noisy_tr).sum())
        w_up = float(np.clip(n_clean / max(1, 4 * len(E_up)), 1, 5)) if len(E_up) else 1.0
        w = np.where(is_up, w_up, 1.0)

        W0, b0 = base.w["head_w"], base.w["head_b"]
        if mode == "train":
            W0, b0 = np.zeros_like(W0), np.zeros_like(b0)
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        steps = 15

        def on_step(step, W, b):
            P_tr, P_va = lite.softmax(E[tr] @ W + b), lite.softmax(E[va] @ W + b)
            history["loss"].append(float(-np.mean(np.log(np.clip(P_tr[np.arange(len(tr)), y[tr]], 1e-12, None)))))
            history["val_loss"].append(float(-np.mean(np.log(np.clip(P_va[np.arange(len(va)), y[va]], 1e-12, None)))))
            history["accuracy"].append(float((P_tr.argmax(1) == y[tr]).mean()))
            history["val_accuracy"].append(float((P_va.argmax(1) == y[va]).mean()))
            progress("training the last layer", epoch=min(step, steps), epochs=steps,
                     loss=history["loss"][-1], accuracy=history["accuracy"][-1],
                     val_loss=history["val_loss"][-1], val_accuracy=history["val_accuracy"][-1])

        progress("training the last layer", epoch=0, epochs=steps)
        W, b = lite.fit_head(E[tr], y[tr], w[tr], W0, b0, max_iter=steps * 10, on_step=on_step, every=10)
        new = base.with_head(W, b)

        progress("testing the new model")
        test_metrics = metrics.from_proba(new.head_proba(E_te), y_te)
        new_after = metrics.accuracy(new.head_proba(E_ck), y_ck) if len(y_ck) else None

        def save(path, info):
            new.save(path)
            with open(str(path)[:-4] + "_info.json", "w") as f:
                json.dump(info, f, indent=2)

        info = {"engine": "lite", "base_version": base_version,
                "data": {"original_train": n_clean, "noisy_copies": int(noisy_tr.sum()),
                         "uploads_learned": int(len(E_up)), "uploads_checked": int(len(E_ck)),
                         "upload_weight": w_up, "train_rows": int(len(tr)), "val_rows": int(len(va))}}
        _finish(run_id, mode, pending_ids, n_new, n_clean, (test_before, new_before),
                (test_metrics["accuracy"], new_after), test_metrics, history, info, save)
    except Exception as exc:
        traceback.print_exc()
        database.update_run(run_id, status="failed", finished_at=registry.now(),
                            message=str(exc)[:500], progress={"stage": "failed"})


def _job_tf(run_id, mode):
    from src import model as M

    def progress(stage, **extra):
        database.update_run(run_id, progress={"stage": stage, **extra})

    try:
        progress("loading the training data")
        X_tr, y_tr, rows_tr = dataset("train")
        X_te, y_te, _ = dataset("test")

        pending_ids = {r["id"] for r in database.pending_uploads()}
        uploads = database.rows("SELECT * FROM uploads ORDER BY id")
        learn = [r for r in uploads if not r["holdout"]]
        check = [r for r in uploads if r["holdout"]]
        n_new = sum(1 for r in learn if r["id"] in pending_ids)
        if mode == "retrain" and n_new == 0:
            raise RuntimeError("no new recordings to learn from; upload some first")

        progress("preparing the uploaded recordings")
        X_up, y_up, learn = _load_uploads(learn) if learn else (X_tr[:0], y_tr[:0], [])
        X_ck, y_ck, check = _load_uploads(check) if check else (X_tr[:0], y_tr[:0], [])

        base_version, base_path, _ = registry.current()
        base = M.load(base_path)
        test_before = M.accuracy(base, X_te, y_te)
        new_before = M.accuracy(base, X_ck, y_ck)

        X = np.concatenate([X_tr, X_up])
        y = np.concatenate([y_tr, y_up])
        # Stratify by speaker for FSDD's six voices; Speech Commands has one
        # clip per speaker, so those count as one group.
        groups = [_group(r["speaker"]) for r in rows_tr] + [f"up_{r['speaker']}" for r in learn]
        # A few uploads would be drowned out by thousands of original clips,
        # so each upload counts more (between 1x and 5x).
        w_up = float(np.clip(len(X_tr) / max(1, 4 * len(X_up)), 1, 5)) if len(X_up) else 1.0
        w = np.concatenate([np.ones(len(X_tr)), np.full(len(X_up), w_up)]).astype(np.float32)
        try:
            tr, va = M.split_train_val(X, y, groups)
        except ValueError:      # a speaker/digit pair with a single file
            tr, va = M.split_train_val(X, y)

        # Noisy copies of half the training part (never of the validation part)
        progress("adding background noise to half the training clips")
        rng = np.random.default_rng(SEED)
        paths = [r["path"] for r in rows_tr] + [r["stored_path"] for r in learn]
        noisy = np.array([i for i in tr if rng.random() < 0.5], dtype=int)
        X_fit = np.concatenate([X[tr], noisy_inputs([paths[i] for i in noisy])])
        y_fit = np.concatenate([y[tr], y[noisy]])
        w_fit = np.concatenate([w[tr], w[noisy]])

        epochs = 15 if mode == "retrain" else 50

        def on_epoch(epoch, logs):
            progress("training", epoch=epoch, epochs=epochs,
                     loss=logs.get("loss"), accuracy=logs.get("accuracy"),
                     val_loss=logs.get("val_loss"), val_accuracy=logs.get("val_accuracy"))

        M.set_seed()
        if mode == "retrain":
            new_model, history = M.fine_tune(base, X_fit, y_fit, X[va], y[va], sample_weight=w_fit,
                                             epochs=epochs, on_epoch=on_epoch, verbose=0)
        else:
            new_model, history = M.train(X_fit, y_fit, X[va], y[va], epochs=epochs,
                                         on_epoch=on_epoch, verbose=0)

        progress("testing the new model")
        test_metrics = M.evaluate(new_model, X_te, y_te)
        test_after = test_metrics["accuracy"]
        new_after = M.accuracy(new_model, X_ck, y_ck)

        ok_test = test_after >= test_before - MAX_TEST_DROP
        ok_new = new_before is None or new_after >= new_before
        promote = ok_test and ok_new

        version = registry.next_version()
        path = registry.path_for(version)
        info = {
            "version": version, "created_at": registry.now(), "mode": mode, "run_id": run_id,
            "base_version": base_version,
            "data": {"original_train": int(len(X_tr)), "uploads_learned": int(len(X_up)),
                     "uploads_checked": int(len(X_ck)), "upload_weight": w_up,
                     "train_rows": int(len(X_fit)), "noisy_copies": int(len(noisy)),
                     "val_rows": int(len(va))},
            "test_metrics": test_metrics,
            "uploads_holdout_accuracy": new_after,
            "history": history,
        }
        M.save(new_model, path, info)
        note = f"{mode} run {run_id}" + ("" if promote else " (not used: not better)")
        registry.add(version, path, note, make_current=promote)

        if pending_ids:
            marks = ",".join(str(i) for i in pending_ids)
            database.execute(f"UPDATE uploads SET used_in_run=? WHERE id IN ({marks})", (run_id,))

        why = []
        if not ok_test:
            why.append(f"test accuracy fell from {test_before:.1%} to {test_after:.1%}")
        if not ok_new:
            why.append(f"held-aside uploads fell from {new_before:.1%} to {new_after:.1%}")
        database.update_run(
            run_id, status="done" if promote else "kept_old", finished_at=registry.now(),
            new_version=version, promoted=int(promote), n_new=n_new, n_replay=int(len(X_tr)),
            epochs=len(history.get("loss", [])), test_acc_before=test_before,
            test_acc_after=test_after, new_acc_before=new_before, new_acc_after=new_after,
            metrics={"test": test_metrics, "history": history},
            message=(f"version {version} is live" if promote
                     else "kept the old model: " + "; ".join(why)),
            progress={"stage": "finished"})
    except Exception as exc:
        traceback.print_exc()
        database.update_run(run_id, status="failed", finished_at=registry.now(),
                            message=str(exc)[:500], progress={"stage": "failed"})
