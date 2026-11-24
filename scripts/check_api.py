"""End-to-end check of a running API: predict, upload, trigger, retrain, evaluate.

Start the API on an empty storage folder first, for example:
    $env:STORAGE_DIR = "$env:TEMP\\voice-digits-check"; uvicorn src.api:app --port 8000
then:
    python scripts/check_api.py --host http://localhost:8000

It uploads 400 of the new voice's recordings (takes 10 to 49), presses
Retrain, waits for the run and checks that the new model is live and better
on the new voice.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--skip-retrain", action="store_true")
    args = ap.parse_args()
    h = args.host.rstrip("/")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    health = requests.get(h + "/health", timeout=30).json()
    check(health["status"] == "ok", f"/health: model v{health['model_version']}, served by {health['served_by']}")
    v0 = health["model_version"]

    wav = ROOT / "data/test/3/3_lucas_2.wav"
    r = requests.post(h + "/predict", files={"file": (wav.name, wav.read_bytes(), "audio/wav")}).json()
    check(r["digit"] == 3, f"/predict {wav.name} -> {r['digit']} ({r['confidence']:.2f}) in {r['latency_ms']} ms")

    r = requests.post(h + "/predict", files={"file": ("silence.wav", b"RIFF0000WAVEfmt ", "audio/wav")})
    check(r.status_code == 422, f"/predict on a broken file -> HTTP {r.status_code}: {r.json().get('detail')}")

    s = requests.get(h + "/sample?dataset=new_voice")
    check(s.ok and s.headers.get("X-True-Label") is not None, f"/sample -> {s.headers.get('X-File-Name')}")

    ins = requests.get(h + "/insights").json()
    check(len(ins["mean_spectrogram"]) == 10, "/insights has 10 mean spectrograms")

    ev = requests.post(h + "/evaluate?dataset=test").json()
    check(ev["accuracy"] > 0.8, f"/evaluate test: accuracy {ev['accuracy']:.3f}, F1 {ev['f1_macro']:.3f}")
    ev_new = requests.post(h + "/evaluate?dataset=new_voice").json()
    print(f"     new voice before retraining: accuracy {ev_new['accuracy']:.3f}")

    # bulk upload: takes 10-49 of the new voice, as a few batches
    files = sorted(p for p in (ROOT / "data/new_voice").glob("*/*.wav") if int(p.stem.split("_")[2]) >= 10)
    random.Random(1).shuffle(files)
    batches = [files[i:i + 100] for i in range(0, len(files), 100)]
    last = None
    for b in batches:
        payload = [("files", (p.name, p.read_bytes(), "audio/wav")) for p in b]
        last = requests.post(h + "/upload", files=payload, data={"label": "auto", "speaker": ""}).json()
        print(f"     uploaded {last['saved']} (rejected {len(last['rejected'])}, duplicates {last['duplicates']}), "
              f"batch accuracy {last['batch_accuracy']:.2f}")
    check(last["trigger"]["fired"], "trigger fired: " + "; ".join(last["trigger"]["reasons"]))

    again = requests.post(h + "/upload", files=[("files", (files[0].name, files[0].read_bytes(), "audio/wav"))],
                          data={"label": "auto"}).json()
    check(again["duplicates"] == 1, "uploading the same file twice is caught as a duplicate")

    if args.skip_retrain:
        sys.exit(0 if ok else 1)

    run = requests.post(h + "/retrain", data={"mode": "retrain"}).json()
    print(f"     retraining run {run['run_id']} started")
    busy = requests.post(h + "/retrain", data={"mode": "retrain"})
    check(busy.status_code == 409, "a second retrain while one runs is refused (HTTP 409)")
    t0 = time.time()
    while True:
        st = requests.get(h + "/retrain/status").json()["latest"]
        if st["status"] != "running":
            break
        p = st.get("progress") or {}
        print(f"     {time.time() - t0:5.0f}s  {p.get('stage')}  epoch {p.get('epoch', '-')}/{p.get('epochs', '-')}")
        time.sleep(10)
    print(f"     finished in {time.time() - t0:.0f}s: {st['status']}: {st['message']}")
    check(st["status"] == "done", "retrained model promoted")
    check(st["test_acc_after"] >= st["test_acc_before"] - 0.02,
          f"test accuracy {st['test_acc_before']:.3f} -> {st['test_acc_after']:.3f}")
    check(st["new_acc_after"] >= st["new_acc_before"],
          f"held-aside uploads {st['new_acc_before']:.3f} -> {st['new_acc_after']:.3f}")

    health = requests.get(h + "/health").json()
    check(health["model_version"] == v0 + 1, f"live model is now v{health['model_version']}")
    ev_new2 = requests.post(h + "/evaluate?dataset=new_voice").json()
    print(f"     new voice after retraining: accuracy {ev_new2['accuracy']:.3f} "
          f"(all 500 clips, 400 of which it trained on)")
    ev_up = requests.post(h + "/evaluate?dataset=uploads").json()
    print(f"     held-aside uploads after retraining: accuracy {ev_up['accuracy']:.3f} on {ev_up['n']} clips")
    trig = requests.get(h + "/trigger").json()
    check(not trig["fired"], f"trigger calm after retraining ({trig['pending']} waiting)")
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
