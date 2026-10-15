from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field,field_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=16,ge=1)
    temperature: float = Field(default=1.0,ge=0.0,le=2)
    top_p: float = Field(default=1.0,gt=0,le=1)
    stream:bool = Field(default=False)
    stream_options:StreamOptions|None = None
    ignore_eos:bool = False


FinishReason = Literal["stop", "length"]


def new_id(prefix: str = "cmpl") -> str:
    """OpenAI-style id like 'cmpl-9f3c...'."""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: FinishReason | None = None


class CompletionResponse(BaseModel):
    """Used for the full (non-stream) response AND for each stream chunk."""
    id: str
    object: Literal["text_completion"] = "text_completion"
    created:int=Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


SSE_DONE = "data: [DONE]\n\n"


def sse_event(payload: BaseModel) -> str:
    """One model -> one SSE event: 'data: <json>' followed by a blank line."""
    return f"data: {payload.model_dump_json(exclude_none=True)}\n\n"


RequestStatus = Literal["ok", "error", "rejected", "timeout"]


class RequestRecord(BaseModel):
    """
    Raw timestamps for one request, all from time.perf_counter() in one process.
    """
    request_id: str
    stack: str
    prompt_len: int
    max_tokens: int

    t_arrival: float
    t_send: float
    chunk_times: list[float] = Field(default_factory=list)
    t_end: float | None = None

    status: RequestStatus = "ok"
    http_status: int | None = None
    error: str | None = None

    @computed_field
    @property
    def ttft(self) -> float | None:
        """Time to first token = first chunk - send."""
        if not self.chunk_times:
            return None
        return self.chunk_times[0]-self.t_send

    @computed_field
    @property
    def tpot(self) -> float | None:
        """Time per output token = (last chunk - first chunk) / (n_chunks - 1)."""
        if len(self.chunk_times)<2:
            return None
        return (self.chunk_times[-1]-self.chunk_times[0])/(len(self.chunk_times)-1)

    @computed_field
    @property
    def itl(self) -> list[float]:
        """Inter-token latencies = gaps between consecutive chunks."""
        if len(self.chunk_times)<2:
            return []
        return [j-i for i,j in zip(self.chunk_times,self.chunk_times[1:])]


    @computed_field
    @property
    def e2el(self) -> float | None:
        """End-to-end latency = end of stream - send."""
        if self.t_end is None:
            return None
        return self.t_end-self.t_send

    @computed_field
    @property
    def send_lag(self) -> float:
        """
        How late the client sent vs the schedule.
        """
        return self.t_send-self.t_arrival


Protocol = Literal["openai", "triton_hf"]


class TargetConfig(BaseModel):
    """Which server to hit and how to talk to it."""
    stack:str
    protocol:Protocol
    url:str
    model:str
    stats_path:str | None=None
    @field_validator("url")
    @classmethod
    def strip_url(cls,value:str)->str:
        return value.rstrip("/")


class TraceConfig(BaseModel):
    """The workload. Same values + same seed -> identical trace for every stack."""
    num_requests: int= Field(ge=1)
    prompt_len:int=Field(ge=1)
    output_dist:Literal["constant","exponential"]
    output_mean:int=Field(ge=1)
    output_max:int=Field(ge=1)
    seed:int=0

class ExperimentConfig(BaseModel):

    """One experiment run = one target x several arrival rates."""

    name:str
    target:TargetConfig
    trace:TraceConfig
    rates:list[float]
    warmup_requests:int=5
    request_timeout_s:float=600.0 
    stats_period_s:float=1.0 


