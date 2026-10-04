"""HTTP boundary with payload validation and Prometheus instrumentation."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app

from rcs.preprocess.image import decode_image
from rcs.settings import Settings

app = FastAPI(title="Realtime CV Serving")
settings = Settings()
requests = Counter("rcs_requests_total", "Requests", ["status"])
latency = Histogram("rcs_request_seconds", "End-to-end latency")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    raise HTTPException(503, "model not loaded")


@app.post("/v1/detect")
async def detect(request: Request, image: UploadFile = File(...)) -> dict[str, object]:
    with latency.time():
        try:
            decoded = decode_image(
                await image.read(), settings.max_upload_bytes, settings.max_image_pixels
            )
        except ValueError as error:
            requests.labels("400").inc()
            raise HTTPException(400, str(error)) from error
        requests.labels("503").inc()
        raise HTTPException(
            503, f"model unavailable; accepted image {decoded.shape[1]}x{decoded.shape[0]}"
        )


app.mount("/metrics", make_asgi_app())
