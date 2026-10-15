from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendConfig(BaseModel):
    """Where the gateway forwards requests. Plain BaseModel: it's nested in GatewaySettings."""

    url: str = "http://localhost:8000"
    model: str = "NousResearch/Llama-3.2-1B"
    timeout_s: float = Field(300.0, gt=0)

    @field_validator("url")
    @classmethod
    def strip_slash(cls,v:str)->str:
        return v.rstrip("/")

class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GW_",env_nested_delimiter="__",env_file=".env",extra="ignore"
    )

    host: str = "0.0.0.0"
    port: int = Field(8080, ge=1, le=65535)

    
    max_inflight:int=Field(default=64,ge=1)
    max_queue:int=Field(default=256,ge=0)
    queue_timeout_s:float=Field(default=10.0,gt=0)

    backend: BackendConfig = Field(default_factory=BackendConfig)

