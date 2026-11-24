"""Flood the API with prediction requests.

Each simulated user picks a random test recording and posts it to /predict,
waits 0.1 to 0.5 s, and repeats. One in ten requests is a /health check.

    locust -f locust/locustfile.py --host http://localhost:8080
    (then open http://localhost:8089, or add --headless -u 100 -r 10 -t 60s)
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parent.parent
FILES = [(p.name, p.read_bytes()) for p in sorted((ROOT / "data" / "test").glob("*/*.wav"))]


class AppUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        name, data = random.choice(FILES)
        with self.client.post("/predict", files={"file": (name, data, "audio/wav")},
                              name="POST /predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif r.json().get("digit") is None:
                r.failure("no digit in the answer")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
