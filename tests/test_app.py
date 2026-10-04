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
    assert response.status_code == 400
