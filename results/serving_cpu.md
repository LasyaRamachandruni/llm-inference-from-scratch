gpt2 (random (timing does not depend on weights)), Intel(R) Xeon(R) Processor @ 2.80GHz, 2 vCPUs, 2 torch threads, torch 2.14.1+cu130.
64 requests, outputs log-uniform 16-256 tokens, arrivals: all at t=0; 16 slots. Mean [95% CI] over 3 runs.

| Scheduler | Output tokens/s | Mean latency (s) | p99 latency (s) | Mean TTFT (s) | Wasted rows |
|---|---:|---:|---:|---:|---:|
| static | 44.3 [44.0, 44.5] | 61.69 [61.56, 61.87] | 140.88 [140.30, 141.78] | 48.47 [48.28, 48.70] | 53% |
| continuous | 84.6 [83.3, 86.3] | 37.79 [36.74, 38.58] | 74.62 [73.13, 75.63] | 22.01 [21.23, 22.61] | 0% |

Continuous / static throughput: 1.91x.
