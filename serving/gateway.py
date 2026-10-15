"""
Async gateway in front of an OpenAI-compatible backend (stack C).

  POST /v1/completions   validate -> admission control -> forward -> stream back
  GET  /health           liveness
  GET  /stats            inflight / waiting / admitted / rejected counters

Settings come from env vars (common/settings.py, prefix GW_), e.g.
  GW_BACKEND__URL=http://localhost:8000
  GW_BACKEND__MODEL=grep -rn "meta-llama" --exclude-dir=third_party .
  GW_MAX_INFLIGHT=32  GW_MAX_QUEUE=64  GW_QUEUE_TIMEOUT_S=10

Run:  python -m serving.gateway
"""
import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse

from common.schemas import CompletionRequest
from common.settings import GatewaySettings

settings = GatewaySettings()

slots = asyncio.Semaphore(settings.max_inflight)
state = {"inflight": 0, "waiting": 0, "admitted": 0,
         "rejected_queue_full": 0, "rejected_timeout": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client=httpx.AsyncClient(
        timeout=httpx.Timeout(settings.backend.timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="llm-gateway", lifespan=lifespan)


def reject(reason: str) -> JSONResponse:
    """Fast 503: the client should retry later."""
    return JSONResponse(status_code=503, content={"error": reason}, headers={"Retry-After": "1"})


async def acquire_slot() -> str | None:
    """Admit the request (return None) or return the reason it was rejected."""
    if slots.locked() and state["waiting"] >= settings.max_queue:
        state["rejected_queue_full"] += 1
        return "queue full"

    state["waiting"]+=1
    try:
        await asyncio.wait_for(slots.acquire(), timeout=settings.queue_timeout_s)
    except asyncio.TimeoutError:
        state["rejected_timeout"]+=1
        return "queue timeout"
    finally:
        state["waiting"]-=1

    state["inflight"]+=1
    state["admitted"]+=1
    return None


def release_slot() -> None:
    state["inflight"]-=1
    slots.release()


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    reason = await acquire_slot()
    if reason is not None:
        return reject(reason)

    body = req.model_dump(exclude_none=True)
    body["model"] = settings.backend.model
    url = f"{settings.backend.url}/v1/completions"
    client: httpx.AsyncClient = app.state.client

    if not req.stream:
        try:
            resp = await client.post(url, json=body)
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")
        except httpx.HTTPError as e:
            return JSONResponse(status_code=502, content={"error": f"backend error: {e}"})
        finally:
            release_slot()

    try:
        upstream = await client.send(client.build_request("POST", url, json=body), stream=True)
    except httpx.HTTPError as e:
        release_slot()
        return JSONResponse(status_code=502, content={"error": f"backend error: {e}"})

    if upstream.status_code != 200:
        content = await upstream.aread()
        await upstream.aclose()
        release_slot()
        return Response(content=content, status_code=upstream.status_code,
                        media_type="application/json")

    async def passthrough():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            release_slot()

    return StreamingResponse(passthrough(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return {**state, "max_inflight": settings.max_inflight, "max_queue": settings.max_queue}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)