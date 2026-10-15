import argparse
import asyncio
import json
import time
from pathlib import Path
from collections import Counter
import httpx
import numpy as np
import yaml
from common.schemas import ExperimentConfig,RequestRecord,TargetConfig
from bench.trace import TraceRequest,build_trace

def load_config(path: str) -> ExperimentConfig:
    """YAML file -> validated ExperimentConfig."""
    with open(path,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)


def build_request(target: TargetConfig, req: TraceRequest) -> tuple[str, dict]:
    """Return (full URL, JSON body) for one request, per target.protocol."""
    if target.protocol=="openai":
        url=f"{target.url}/v1/completions"
        body={
            "model":target.model,
            "prompt":req.prompt,
            "max_tokens":req.max_tokens,
            "temperature":0,
            "ignore_eos":True,
            "stream":True

        }
    elif target.protocol=="triton_hf":
        url=f"{target.url}/v2/models/{target.model}/generate_stream"
        body={
            "text_input":req.prompt,
            "parameters":{
                "max_tokens":req.max_tokens,
                "temperature":0.0,
                "ignore_eos":True,
            }

        }
    return url,body


    
    


    

def parse_sse_line(line: str, protocol: str) -> str | None:
    """
    One line of the stream -> the text it carries.
      returns None  for lines without data (blank lines, comments, "data: [DONE]")
      returns ""    for data chunks with no text (e.g. the final finish_reason chunk)
      raises RuntimeError if the chunk contains an "error" key (Triton can send this with HTTP 200)
    """
    
    line=line.strip()
    if not line.startswith("data:"):
        return None

    payload = line[len("data:"):].strip()

    if not payload or payload=="[DONE]":
        return None

    obj=json.loads(payload)
    
    if "error" in obj:
        raise RuntimeError(str(obj["error"]))


    if protocol=="openai":
        choices=obj.get("choices")
        return choices[0].get("text","") if choices else ""
    

    return obj.get("text_output", "")

    


async def send_one(client: httpx.AsyncClient, target: TargetConfig,
                   req: TraceRequest, t0: float) -> RequestRecord:
    """
    Sleep until t0 + req.t_arrival, send the request, timestamp every non-empty text chunk.
    Never raises: every outcome becomes a RequestRecord with a status.
    """

    t_arrival=t0 + req.t_arrival
    await asyncio.sleep(max(0.0, t_arrival - time.perf_counter()))
    url,body=build_request(target,req)

    record=RequestRecord(
        request_id=req.request_id,
        stack=target.stack,
        prompt_len=req.prompt_len,
        max_tokens=req.max_tokens,
        t_arrival=t_arrival,
        t_send=time.perf_counter(),
    )


    try:
        async with client.stream("POST",url,json=body) as resp:
            record.http_status=resp.status_code
            if resp.status_code!=200:
                response_body=await resp.aread()
                record.error=response_body.decode(
                    "utf-8", errors="replace"
                )[:500]
                record.status =(
                    "rejected"
                    if resp.status_code in (429, 503)
                    else "error"
                )
            else:
                async for line in resp.aiter_lines():
                    text=parse_sse_line(line,target.protocol)
                    if text:
                        record.chunk_times.append(time.perf_counter())
                record.status="ok"

    except httpx.TimeoutException as e:
        record.status="timeout"
        record.error=str(e)

    except Exception as e:
        record.status="error"
        record.error=str(e)

    finally:
        record.t_end=time.perf_counter()

    return record


    


async def poll_stats(client: httpx.AsyncClient, target: TargetConfig, period_s: float,
                     stop: asyncio.Event, out: list[dict]) -> None:
    """Every period_s, GET target.url + target.stats_path and append {"t": ..., "data": ...} to out."""

    url=target.url+(target.stats_path or "")
    while not stop.is_set():
        try:
            resp=await client.get(url)
            if resp.status_code==200:
                out.append({"t":time.perf_counter(), "data":resp.json()})
        except Exception:
            pass
        await asyncio.sleep(period_s)
    


async def run_rate(cfg: ExperimentConfig, rate: float) -> tuple[list[RequestRecord], list[dict]]:
    """Warmup, then the full open-loop trace at this rate. Returns (records, stats snapshots)."""
    
    trace=build_trace(cfg.trace,rate)
    snapshots: list[dict]=[]
    timeout = httpx.Timeout(cfg.request_timeout_s, connect=10.0)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for req in trace[:cfg.warmup_requests]:
            warmup_req=req.model_copy(update={"t_arrival":0.0})
            await send_one(client, cfg.target, warmup_req, t0=time.perf_counter())

        poller = None
        stop = asyncio.Event()

        if cfg.target.stats_path:
            period_s = getattr(cfg, "stats_period_s", 1.0)
            poller = asyncio.create_task(
                poll_stats(client, cfg.target, period_s, stop, snapshots)
            )

        t0 = time.perf_counter()
        records = await asyncio.gather(
            *(send_one(client, cfg.target, r, t0) for r in trace)
        )

        if poller is not None:
            stop.set()
            await poller

        return list(records),snapshots
def write_jsonl(rows: list, path: Path) -> None:
    """Write pydantic models (model_dump_json) or dicts (json.dumps), one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            line=row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(row)
            f.write(line + "\n")


def summarize(records: list[RequestRecord]) -> str:
    """One line for the terminal: counts per status, TTFT p50/p99, mean TPOT, max send_lag."""
    counts=Counter(r.status for r in records)
    count_str=" ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    max_lag_ms=max((r.send_lag for r in records), default=0.0) * 1000

    ok=[r for r in records if r.status == "ok" and r.ttft is not None]
    if not ok:
        return f"{count_str} | no successful requests | max send_lag {max_lag_ms:.1f} ms"

    ttft_ms=np.array([r.ttft for r in ok]) * 1000
    tpots=[r.tpot for r in ok if r.tpot is not None]
    tpot_ms=float(np.mean(tpots)) * 1000 if tpots else float("nan")

    line=(
        f"{count_str} | TTFT p50 {np.percentile(ttft_ms, 50):.1f} ms, "
        f"p99 {np.percentile(ttft_ms, 99):.1f} ms | TPOT mean {tpot_ms:.2f} ms | "
        f"max send_lag {max_lag_ms:.1f} ms"
    )
    if max_lag_ms > 50:
        line+= "  WARNING: client fell behind schedule"
    return line


async def main_async(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = Path("results/raw") / cfg.name
    for rate in cfg.rates:
        records, stats = await run_rate(cfg, rate)
        write_jsonl(records, out_dir / f"rate_{rate}.jsonl")
        if stats:
            write_jsonl(stats, out_dir / f"rate_{rate}_stats.jsonl")
        print(f"[{cfg.name}] rate={rate}: {summarize(records)}")


def main() -> None:
    parser=argparse.ArgumentParser(description="Open-loop load generator")
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    args=parser.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()