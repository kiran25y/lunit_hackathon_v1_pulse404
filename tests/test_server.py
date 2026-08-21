from fastapi.testclient import TestClient
import subprocess
import sys

from app.server import (
    DRIVER_MODEL_NAME,
    app,
)


client = TestClient(app)


def test_server_import_does_not_load_mcp_stack():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.server; "
                "assert 'mcp' not in sys.modules; "
                "assert 'app.mcp_client' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models():
    response = client.get("/v1/models")

    assert response.status_code == 200

    body = response.json()

    assert body["object"] == "list"

    assert (
        body["data"][0]["id"]
        == DRIVER_MODEL_NAME
    )


def test_empty_messages_rejected():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": DRIVER_MODEL_NAME,
            "messages": [],
        },
    )

    assert response.status_code == 400


def test_streaming_rejected():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": DRIVER_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "stream": True,
        },
    )

    assert response.status_code == 400
