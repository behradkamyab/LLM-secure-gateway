import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from gateway.common.schema import GatewayResponse, HealthResponse, RedactionRequest
from gateway.infer.engine import GatewayEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = os.getenv("GATEWAY_CONFIG", "configs/e0_zeroshot.yaml")
    logger.info("Loading GatewayEngine from config: %s", config_path)
    app.state.engine = GatewayEngine(config_path=config_path)
    logger.info("GatewayEngine loaded and ready.")
    try:
        yield
    finally:
        logger.info("Shutting down GatewayEngine.")
        app.state.engine = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="AI Security Gateway",
    description="Semantic routing and hybrid PII redaction gateway.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "engine", None) is not None
    return HealthResponse(status="ok" if loaded else "loading", model_loaded=loaded)


@app.post("/api/v1/gateway", response_model=GatewayResponse)
async def process_gateway_request(
    payload: RedactionRequest,
    request: Request,
) -> GatewayResponse:
    engine: GatewayEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Gateway engine is not ready yet.")

    start = time.perf_counter()
    result = await run_in_threadpool(engine.process_request, payload.text)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Processed request route=%s latency_ms=%.2f",
        result["route"]["destination"],
        latency_ms,
    )
    return GatewayResponse.model_validate(result)
