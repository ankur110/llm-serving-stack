import numpy as np
from pydantic import BaseModel
from common.schemas import TraceConfig

class TraceRequest(BaseModel):
    """One request in the workload."""
    request_id:str
    t_arrival:float
    prompt:str
    prompt_len:int
    max_tokens:int


def poisson_arrivals(n: int, rate: float, rng: np.random.Generator) -> list[float]:
    """n arrival times (seconds) of a Poisson process with `rate` req/s."""
    gaps=rng.exponential(scale=1.0/rate,size=n)
    times=np.cumsum(gaps)
    return times.toist()


def output_lengths(cfg: TraceConfig, rng: np.random.Generator) -> list[int]:
    """Output length per request: constant, or exponential truncated by REJECTION."""
    if cfg.output_dist == "constant":
        return [cfg.output_mean] * cfg.num_requests
    lengths=[]
    while len(lengths) < cfg.num_requests:
        sample=max(
            1,
            int(np.ceil(rng.exponential(scale=cfg.output_mean))),
        )

        if sample<=cfg.output_max:
            lengths.append(sample)

    return lengths


def make_prompt(i: int, prompt_len: int) -> str:
    """Unique prompt of about `prompt_len` tokens: the request number + repeated ' hello'."""
    return f"{i}"+ "hello"*prompt_len


def build_trace(cfg: TraceConfig, rate: float) -> list[TraceRequest]:
    """The full workload for one arrival rate."""
    rng=np.random.default_rng(cfg.seed)
    arrivals=poisson_arrivals(n=cfg.num_requests,rate=rate,rng=rng)
    lengths = output_lengths(cfg, rng)
    trace = [
        TraceRequest(
            request_id=f"r{i:05d}",
            t_arrival=arrivals[i],
            prompt=make_prompt(i, cfg.prompt_len),
            prompt_len=cfg.prompt_len,
            max_tokens=lengths[i],
        )
        for i in range(cfg.num_requests)
    ]
    return trace


if __name__ == "__main__":
    cfg = TraceConfig(
        num_requests=2000,
        prompt_len=16,
        output_dist="exponential",
        output_mean=128,
        output_max=1024,
        seed=0,
    )

    trace = build_trace(cfg, rate=4.0)

    duration = trace[-1].t_arrival
    measured_rate = len(trace) / duration

    print(f"Duration: {duration:.2f} seconds")
    print(f"Measured arrival rate: {measured_rate:.2f} req/s")
    print(
        "Mean output length:",
        np.mean([r.max_tokens for r in trace]),
    )
    print(
        "Maximum output length:",
        max(r.max_tokens for r in trace),
    )
    print("First prompt:", repr(trace[0].prompt[:40]))

    trace_again = build_trace(cfg, rate=4.0)
    assert trace == trace_again
    print("Determinism check: passed")

    constant_cfg = TraceConfig(
        num_requests=100,
        prompt_len=16,
        output_dist="constant",
        output_mean=32,
        output_max=1024,
        seed=0,
    )

    constant_trace = build_trace(constant_cfg, rate=4.0)
    assert all(r.max_tokens == 32 for r in constant_trace)
    print("Constant-length check: passed")