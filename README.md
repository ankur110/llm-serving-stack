# LLM Serving Stack

Two ways to serve the same model (Llama-3.2-1B, bf16, A100), benchmarked under the same open-loop load. The goal is to show how much the serving engine decides latency and throughput, and what an admission-control gateway adds on top.

## Stacks

**A: static batching.** A Hugging Face transformers engine behind Triton's Python backend. Triton groups requests into a batch (up to 32), and the batch runs until its longest request finishes. New requests wait for the whole batch to finish.

**C: TensorRT-LLM.** `trtllm-serve` with in-flight (continuous) batching, chunked prefill and a paged KV cache. Requests join and leave the batch at every decode step.

**Gateway.** A small async FastAPI service in front of C. It caps the number of in-flight requests, queues a limited number more and rejects the rest, so latency stays bounded under overload.

## Benchmark

Each run sends 300 requests with 256-token prompts and exponentially distributed outputs (mean 128, cap 1024), arriving as a Poisson process at a fixed rate. Every token's arrival time is recorded. Goodput counts the requests per second that meet both SLOs: TTFT ≤ 1 s and TPOT ≤ 50 ms.

## Results

**E1: A vs C**

| Rate (req/s) | A TTFT p50 | C TTFT p50 | A goodput | C goodput |
|---|---|---|---|---|
| 1 | 1.93 s | 0.02 s | 0.26 | 0.86 |
| 2 | 2.99 s | 0.02 s | 0.23 | 1.73 |
| 4 | 7.82 s | 0.02 s | 0.12 | 3.44 |

A saturates at about 3 req/s and its queueing delay grows quickly. C keeps TTFT around 20–30 ms up to 32 req/s, peaks at about 37 req/s, and gets about 7x lower TPOT (3 ms vs 16–20 ms).

**E3: C vs gateway → C**

| Rate (req/s) | C TTFT p99 | Gateway TTFT p99 | C goodput | Gateway goodput | Gateway rejected |
|---|---|---|---|---|---|
| 64 | 1.56 s | 1.15 s | 20.6 | 29.0 | 0 |
| 96 | 2.95 s | 1.98 s | 14.6 | 17.0 | 61 / 300 |

Below saturation the gateway adds almost no latency. Above it, the gateway rejects excess load instead of queueing it, which lowers tail TTFT and raises goodput.

Plots are in `results/E1/` and `results/E3/`.
