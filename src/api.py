"""Step 6: the API (FastAPI). It also serves the web dashboard in ui/.

Run locally:  uvicorn src.api:app --port 8000
Docs:         http://localhost:8000/docs
"""
import json
import os
import random
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import database, registry, retrain
from src.config import NEW_VOICE_DIR, RESULTS_DIR, ROOT, TEST_DIR
from src.metrics import from_proba
from src.prediction import Predictor
from src.preprocessing import AudioError, list_folder

UI_DIR = ROOT / "ui"
SERVED_BY = os.environ.get("CONTAINER_NAME") or socket.gethostname()
STARTED_AT = time.time()
state = {"predictor": None, "requests": 0, "predictions": 0}
# How many predictions run at the same time in one container. More than the
# container has CPUs only adds switching between threads; the rest wait in line.
PREDICT_SLOTS = threading.BoundedSemaphore(int(os.environ.get("PREDICT_CONCURRENCY", "2")))


@asynccontextmanager
async def lifespan(app):
    database.init()
    state["predictor"] = Predictor()
    # Run one prediction now so the first user doesn't wait for warm-up.
    warm = next(iter(sorted(TEST_DIR.glob("*/*.wav"))), None)
    if warm:
        state["predictor"].predict_file(str(warm))
    yield


app = FastAPI(title="Voice Digits API",
              description="Spoken digit recognition for voice-first mobile money.",
              version="1.0", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request, call_next):
    state["requests"] += 1
    response = await call_next(request)
    response.headers["X-Served-By"] = SERVED_BY
    return response


def predictor() -> Predictor:
    p = state["predictor"]
    p.reload_if_changed()
    return p


# ---------------------------------------------------------------- health

@app.get("/health")
def health():
    p = predictor()
    return {
        "status": "ok",
        "served_by": SERVED_BY,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(STARTED_AT)),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "model_version": p.version,
        "engine": p.engine,
        "model_uptime_seconds": round(time.time() - p.loaded_at, 1),
        "requests_served": state["requests"],
        "predictions_served": state["predictions"],
    }


# ---------------------------------------------------------------- predict

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = file.file.read()
    if not data:
        raise HTTPException(400, "empty file")
    try:
        with PREDICT_SLOTS:
            result = predictor().predict_file(data)
    except AudioError as exc:
        raise HTTPException(422, f"{file.filename}: {exc}")
    latency = (time.perf_counter() - t0) * 1000
    state["predictions"] += 1
    database.log_prediction(result["digit"], result["confidence"], latency,
                            result["model_version"], SERVED_BY)
    return {**result, "file": file.filename, "latency_ms": round(latency, 1), "served_by": SERVED_BY}


@app.get("/sample")
def sample(dataset: str = Query("test", pattern="^(test|new_voice)$")):
    """A random labelled recording, so the demo can be tried without files."""
    rows = list_folder(TEST_DIR if dataset == "test" else NEW_VOICE_DIR)
    if not rows:
        raise HTTPException(404, "no files")
    r = random.choice(rows)
    return FileResponse(r["path"], media_type="audio/wav",
                        headers={"X-True-Label": str(r["label"]), "X-File-Name": Path(r["path"]).name})


# ---------------------------------------------------------------- model + evaluation

@app.get("/model")
def model_info():
    p = predictor()
    current, versions = registry.versions()
    return {"version": p.version, "engine": p.engine, "info": p.info, "current": current, "versions": versions}


@app.post("/model/activate")
def activate(version: int = Form(...)):
    """Roll back (or forward) to any saved version."""
    try:
        registry.set_current(version)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    predictor()
    return {"version": version}


@app.post("/evaluate")
def evaluate_live(dataset: str = Query("test", pattern="^(test|unseen|new_voice|uploads)$")):
    """Score the live model on a labelled set, as it runs in production."""
    p = predictor()
    t0 = time.perf_counter()
    if dataset == "uploads":
        rows = database.rows("SELECT * FROM uploads WHERE holdout=1")
        if not rows:
            raise HTTPException(400, "no held-aside uploads yet")
        X, y, _ = retrain._load_uploads(rows)
        proba = p.predict_inputs(X)
    else:   # "unseen" = test clips from Speech Commands speakers not in training
        proba, y = retrain.dataset_proba(dataset, p)
    metrics = from_proba(proba, y)
    metrics["seconds"] = round(time.perf_counter() - t0, 2)
    database.execute("INSERT INTO evaluations (at, model_version, dataset, metrics) VALUES (?,?,?,?)",
                     (registry.now(), p.version, dataset, json.dumps(metrics)))
    return {"dataset": dataset, "model_version": p.version, "served_by": SERVED_BY, **metrics}


@app.get("/evaluations")
def evaluations(limit: int = 20):
    out = database.rows("SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (limit,))
    for r in out:
        m = json.loads(r.pop("metrics"))
        r.update({k: m.get(k) for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "n")})
    return out


# ---------------------------------------------------------------- data

@app.get("/insights")
def insights():
    path = RESULTS_DIR / "insights.json"
    if not path.exists():
        raise HTTPException(404, "run `python -m src.insights` first")
    with open(path) as f:
        data = json.load(f)
    uploaded = [0] * 10
    for r in database.rows("SELECT label, COUNT(*) AS n FROM uploads GROUP BY label"):
        uploaded[r["label"]] = r["n"]
    data["counts"]["uploads"] = uploaded
    return data


@app.post("/upload")
def upload(files: List[UploadFile] = File(...), label: str = Form("auto"), speaker: str = Form("")):
    """Bulk upload: many .wav files or a .zip. The digit comes from `label`,
    or from each file name when label is 'auto' (e.g. 7_amina_3.wav)."""
    if label not in ("auto", "") and label not in [str(d) for d in range(10)]:
        raise HTTPException(400, "label must be 0-9 or auto")
    payload = [(f.filename or "file.wav", f.file.read()) for f in files]
    return retrain.save_batch(payload, label=label, speaker=speaker.strip().lower() or None,
                              predictor=predictor())


@app.get("/uploads")
def uploads():
    totals = database.one(
        "SELECT COUNT(*) AS total, SUM(used_in_run IS NULL) AS waiting, SUM(holdout) AS holdout "
        "FROM uploads")
    batches = database.rows("SELECT * FROM batches WHERE n_saved > 0 OR n_rejected > 0 "
                            "ORDER BY id DESC LIMIT 10")
    return {"totals": {k: (v or 0) for k, v in totals.items()}, "batches": batches}


# ---------------------------------------------------------------- training

@app.get("/trigger")
def trigger():
    return retrain.check_trigger(predictor())


@app.post("/retrain")
def start_retrain(mode: str = Form("retrain")):
    if mode not in ("retrain", "train"):
        raise HTTPException(400, "mode must be retrain or train")
    trig = retrain.check_trigger(predictor())
    reason = "button" + (": " + "; ".join(trig["reasons"]) if trig["reasons"] else "")
    run_id = retrain.start(mode, reason, predictor())
    if run_id is None:
        raise HTTPException(409, "a training run is already going")
    return {"run_id": run_id, "mode": mode}


@app.get("/retrain/status")
def retrain_status(limit: int = 8):
    runs = [database.run_dict(r) for r in
            database.rows("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))]
    for r in runs[1:]:
        if r.get("metrics"):
            r["metrics"] = {"history": r["metrics"].get("history")}
    return {"latest": runs[0] if runs else None, "runs": runs}


# ---------------------------------------------------------------- monitoring

@app.get("/stats")
def stats():
    summary = database.one(
        "SELECT COUNT(*) AS n, AVG(latency_ms) AS avg_ms, MAX(at) AS last_at FROM predictions")
    recent = database.rows("SELECT at, latency_ms, digit, confidence, served_by FROM predictions "
                           "ORDER BY id DESC LIMIT 60")
    by_container = database.rows("SELECT served_by, COUNT(*) AS n, AVG(latency_ms) AS avg_ms "
                                 "FROM predictions GROUP BY served_by ORDER BY n DESC LIMIT 10")
    by_digit = database.rows("SELECT digit, COUNT(*) AS n FROM predictions GROUP BY digit")
    return {"summary": summary, "recent": recent[::-1], "by_container": by_container,
            "by_digit": by_digit}


@app.get("/load-tests")
def load_tests():
    path = RESULTS_DIR / "load_tests" / "summary.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- web page

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(UI_DIR / "index.html")


@app.exception_handler(Exception)
async def unhandled(request, exc):
    return JSONResponse(status_code=500, content={"detail": f"{exc.__class__.__name__}: {exc}"})
