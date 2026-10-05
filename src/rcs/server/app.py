"""HTTP API with safe decoding, bounded batching, and model readiness."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app

from rcs.batching.dynamic import DynamicBatcher
from rcs.preprocess.image import decode_image, letterbox
from rcs.runtime.onnx import OnnxModel
from rcs.settings import Settings

settings = Settings()
requests = Counter("rcs_requests_total", "Requests", ["status"])
latency = Histogram("rcs_request_seconds", "End-to-end latency")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    if settings.model_path:
        model = OnnxModel(settings.model_path)

        async def infer(batch: list[np.ndarray]) -> list[np.ndarray]:
            return await asyncio.to_thread(model.infer, batch)

        app.state.batcher = DynamicBatcher(infer, capacity=settings.queue_capacity)
        await app.state.batcher.start()
        app.state.ready = True
    yield
    if getattr(app.state, "batcher", None):
        await app.state.batcher.close()


app = FastAPI(title="Realtime CV Serving", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict[str, str]:
    if not request.app.state.ready:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready"}


@app.post("/v1/detect")
async def detect(
    request: Request, image: UploadFile = File(...), conf: float = 0.25, max_det: int = 100
) -> dict[str, object]:
    if not 0 <= conf <= 1 or not 1 <= max_det <= 300:
        raise HTTPException(422, "invalid detection options")
    if not request.app.state.ready:
        raise HTTPException(503, "model not ready")
    with latency.time():
        try:
            decoded = decode_image(
                await image.read(), settings.max_upload_bytes, settings.max_image_pixels
            )
        except ValueError as error:
            requests.labels("400").inc()
            raise HTTPException(400, str(error)) from error
        tensor, scale, (pad_x, pad_y) = letterbox(decoded, (64, 64))
        try:
            predictions = await request.app.state.batcher.submit(tensor)
        except OverflowError as error:
            raise HTTPException(429, "queue saturated", headers={"Retry-After": "1"}) from error
        detections = [
            {
                "box": [
                    round(float((x1 - pad_x) / scale), 2),
                    round(float((y1 - pad_y) / scale), 2),
                    round(float((x2 - pad_x) / scale), 2),
                    round(float((y2 - pad_y) / scale), 2),
                ],
                "score": round(float(score), 6),
                "class_id": int(class_id),
            }
            for x1, y1, x2, y2, score, class_id in predictions[:max_det]
            if float(score) >= conf
        ]
        requests.labels("200").inc()
        return {"detections": detections, "model": str(settings.model_path)}


app.mount("/metrics", make_asgi_app())
