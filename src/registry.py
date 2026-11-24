"""Which model version is live.

models/digit_cnn.h5 is version 1, trained in the notebook, and never
changes. Every retrain saves a new file in storage/models/ and, if it is good
enough, points registry.json at it. All containers read the same
registry.json, so a model retrained in one container is picked up by all.
"""
import json
import os
import tempfile
from datetime import datetime, timezone

from src.config import BASE_MODEL_INFO, BASE_MODEL_PATH, ENGINE, REGISTRY_PATH, VERSIONS_DIR
from src.lite import BASE_LITE_INFO, BASE_LITE_PATH

# Each engine keeps its own list of versions (an .h5 file can't be loaded by
# the light engine, and the other way round).
if ENGINE == "lite":
    BASE_PATH, BASE_INFO, EXT = BASE_LITE_PATH, BASE_LITE_INFO, ".npz"
    REGISTRY_PATH = REGISTRY_PATH.with_name("registry_lite.json")
else:
    BASE_PATH, BASE_INFO, EXT = BASE_MODEL_PATH, BASE_MODEL_INFO, ".h5"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"current": 1, "versions": [{"version": 1, "path": str(BASE_PATH),
                                        "created_at": None, "note": "first model (notebook)"}]}


def _write(reg):
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=VERSIONS_DIR, suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def current():
    """(version, path, info) of the live model."""
    reg = _read()
    entry = next(v for v in reg["versions"] if v["version"] == reg["current"])
    return entry["version"], entry["path"], load_info(entry["path"])


def stamp():
    """Changes whenever a new model goes live (cheap to check per request)."""
    try:
        return os.stat(REGISTRY_PATH).st_mtime_ns
    except FileNotFoundError:
        return 0


def load_info(model_path):
    info_path = os.path.splitext(str(model_path))[0] + "_info.json"
    if os.path.exists(info_path):
        with open(info_path) as f:
            return json.load(f)
    if str(model_path) == str(BASE_PATH) and BASE_INFO.exists():
        with open(BASE_INFO) as f:
            return json.load(f)
    return {}


def next_version():
    return max(v["version"] for v in _read()["versions"]) + 1


def path_for(version):
    return VERSIONS_DIR / f"digit_cnn_v{version}{EXT}"


def add(version, path, note, make_current):
    reg = _read()
    reg["versions"].append({"version": version, "path": str(path),
                            "created_at": now(), "note": note})
    if make_current:
        reg["current"] = version
    _write(reg)


def versions():
    reg = _read()
    return reg["current"], reg["versions"]


def set_current(version):
    reg = _read()
    if not any(v["version"] == version for v in reg["versions"]):
        raise ValueError(f"no model version {version}")
    reg["current"] = version
    _write(reg)
