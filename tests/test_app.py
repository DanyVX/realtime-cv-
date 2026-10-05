import cv2
import numpy as np
from fastapi.testclient import TestClient

from rcs.server.app import app

client = TestClient(app)


def test_health_and_readiness() -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503


def test_detect_rejects_invalid_input() -> None:
    response = client.post(
        "/v1/detect", files={"image": ("bad.bin", b"nope", "application/octet-stream")}
    )
    assert response.status_code == 503


def test_detect_validates_input_and_returns_batched_result() -> None:
    class FakeBatcher:
        async def submit(self, _tensor: np.ndarray) -> np.ndarray:
            return np.array([[8, 8, 48, 48, 0.9, 0]], dtype=np.float32)

    app.state.ready = True
    app.state.batcher = FakeBatcher()
    invalid = client.post("/v1/detect", files={"image": ("bad", b"nope", "image/jpeg")})
    assert invalid.status_code == 400
    _, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    response = client.post("/v1/detect", files={"image": ("x.png", encoded.tobytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["detections"][0]["score"] == 0.9
    app.state.ready = False
