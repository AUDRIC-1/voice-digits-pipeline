"""Run the Locust flood against 1, 2, ... API containers and save the results.

    python scripts/load_test.py --containers 1 2 3 --users 50 100 --seconds 60

For each container count it scales the Docker Compose service, restarts the
load balancer so it sees every container, waits until the API answers, runs
Locust headless, and keeps the POST /predict numbers. The summary goes to
results/load_tests/summary.json (shown on the dashboard), summary.md and
latency.png.

Use --no-docker to test whatever is already running at --host.
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "load_tests"
RAW = OUT / "raw"


def sh(*cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def wait_until_up(host, containers, timeout=600):
    """Wait until /health answers, and (with several containers) until
    answers have come from as many different containers as expected."""
    seen, t0 = set(), time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(host + "/health", timeout=10) as r:
                seen.add(json.load(r)["served_by"])
            if len(seen) >= containers:
                return seen
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit(f"API not ready after {timeout}s (saw {sorted(seen)})")


def wait_until_idle(host, timeout=300):
    """After a run, overloaded containers may still be working through
    queued requests. Wait until three health checks in a row are fast."""
    fast, t0 = 0, time.time()
    while fast < 3 and time.time() - t0 < timeout:
        t = time.perf_counter()
        try:
            urllib.request.urlopen(host + "/health", timeout=10).read()
            fast = fast + 1 if time.perf_counter() - t < 0.3 else 0
        except Exception:
            fast = 0
        time.sleep(1)


class HostWatch(threading.Thread):
    """Samples the laptop's free RAM and (on Windows) paging while Locust runs,
    so a run distorted by memory pressure can be spotted."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.free, self.pages = threading.Event(), [], []

    def run(self):
        import psutil
        while not self.stop.is_set():
            self.free.append(psutil.virtual_memory().available / 1e9)
            if sys.platform == "win32":
                try:
                    out = subprocess.run(["typeperf", r"\Memory\Pages/sec", "-sc", "1"],
                                         capture_output=True, text=True, timeout=10).stdout
                    rows = [l for l in out.splitlines() if l.startswith('"') and "PDH-CSV" not in l]
                    self.pages.append(float(rows[-1].split(",")[-1].strip().strip('"')))
                except Exception:
                    pass
            self.stop.wait(4)

    def summary(self):
        return {"host_min_free_gb": round(min(self.free), 2) if self.free else None,
                "host_avg_pages_per_s": round(sum(self.pages) / len(self.pages)) if self.pages else None}


def run_locust(host, users, seconds, tag):
    RAW.mkdir(parents=True, exist_ok=True)
    prefix = RAW / tag
    watch = HostWatch()
    watch.start()
    sh(sys.executable, "-m", "locust", "-f", str(ROOT / "locust" / "locustfile.py"), "--host", host,
       "--headless", "-u", str(users), "-r", str(max(1, users // 10)), "-t", f"{seconds}s",
       "--csv", str(prefix), "--only-summary", "--exit-code-on-error", "0")  # failures are results, not crashes
    watch.stop.set()
    with open(f"{prefix}_stats.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "POST /predict")
    return {**watch.summary(),
        "requests": int(row["Request Count"]), "failures": int(row["Failure Count"]),
        "rps": float(row["Requests/s"]), "median_ms": float(row["Median Response Time"]),
        "avg_ms": float(row["Average Response Time"]), "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]), "max_ms": float(row["Max Response Time"]),
    }


def write_summary(results, note):
    OUT.mkdir(parents=True, exist_ok=True)
    if results:
        results[0]["note"] = note
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    lines = ["| Containers | Users | Requests | Failures | Req/s | Median (ms) | Average (ms) | p95 (ms) | p99 (ms) | Laptop: min free RAM (GB) / paging (pages/s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['containers']} | {r['users']} | {r['requests']} | {r['failures']} | {r['rps']:.1f} | "
                     f"{r['median_ms']:.0f} | {r['avg_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | "
                     f"{r.get('host_min_free_gb')} / {r.get('host_avg_pages_per_s')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n\n" + note + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        users = sorted({r["users"] for r in results})
        counts = sorted({r["containers"] for r in results})
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        width = 0.8 / len(counts)
        for k, c in enumerate(counts):
            sel = {r["users"]: r for r in results if r["containers"] == c}
            x = np.arange(len(users)) + k * width - 0.4 + width / 2
            ax[0].bar(x, [sel.get(u, {}).get("median_ms", 0) for u in users], width * 0.9, color=colors[k % 4],
                      label=f"{c} container{'s' if c > 1 else ''}")
            ax[1].bar(x, [sel.get(u, {}).get("rps", 0) for u in users], width * 0.9, color=colors[k % 4])
        for a, t, y in [(ax[0], "Median response time", "ms"), (ax[1], "Throughput", "requests per second")]:
            a.set_xticks(range(len(users)), [f"{u} users" for u in users])
            a.set_title(t, fontweight="bold"); a.set_ylabel(y)
            a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e8e6df"); a.set_axisbelow(True)
        ax[0].legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT / "latency.png", dpi=120)
    except Exception as exc:
        print("chart skipped:", exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--containers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--users", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--host", default="http://localhost:8080")
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    results = []
    for c in args.containers:
        if not args.no_docker:
            # fresh containers for every container count, so no run inherits
            # a backlog from the one before
            sh("docker", "compose", "up", "-d", "--force-recreate", "--scale", f"api={c}")
        seen = wait_until_up(args.host, 1 if args.no_docker else c)
        print(f"containers answering: {sorted(seen)}")
        for u in args.users:
            wait_until_idle(args.host)
            r = run_locust(args.host, u, args.seconds, f"c{c}_u{u}")
            r.update({"containers": c, "users": u, "seconds": args.seconds})
            results.append(r)
            print(json.dumps(r))
            write_summary(results, args.note)


if __name__ == "__main__":
    main()
