| Containers | Users | Requests | Failures | Req/s | Median (ms) | Average (ms) | p95 (ms) | p99 (ms) | Laptop: min free RAM (GB) / paging (pages/s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 2017 | 0 | 36.3 | 80 | 164 | 490 | 1900 | 0.86 / 5853 |
| 1 | 50 | 2530 | 0 | 43.5 | 160 | 704 | 3300 | 6700 | 0.81 / 5445 |
| 2 | 20 | 2095 | 0 | 37.1 | 76 | 158 | 500 | 2000 | 0.08 / 7956 |
| 2 | 50 | 3338 | 4 | 58.0 | 110 | 430 | 2100 | 4200 | 0.53 / 1641 |
| 3 | 20 | 2378 | 0 | 42.2 | 72 | 99 | 250 | 710 | 0.15 / 11925 |
| 3 | 50 | 3890 | 1 | 66.3 | 94 | 356 | 1600 | 3400 | 0.04 / 5183 |

Local Docker Desktop (12 CPUs, 3.7 GB VM) on a laptop with 7.7 GB RAM, other apps closed. nginx round robin; each API container capped at 1 CPU and runs at most 2 predictions at once; fresh containers for each container count. Locust users wait 0.1-0.5 s between requests; each request uploads a random test .wav to POST /predict.
