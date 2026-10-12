from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ffmpeg" in data
    assert "venvs" in data
    assert "sfx_library" in data


def test_api_stages():
    response = client.get("/api/stages")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 12
    # Verify stages in order
    keys = [item["key"] for item in data]
    assert keys == [
        "extract_audio",
        "asr",
        "alignment",
        "diarization",
        "segment_merge",
        "translation",
        "dialogue",
        "tts",
        "sfx",
        "ambience",
        "mix",
        "mux",
    ]


def test_api_videos():
    response = client.get("/api/videos")
    assert response.status_code == 200
    data = response.json()
    assert "input" in data
    assert "output" in data
    assert isinstance(data["input"], list)
    assert isinstance(data["output"], list)


def test_api_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_api_dub_status_when_idle():
    response = client.get("/api/dub/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] is False
    assert data["stages"] == {}
    assert data["overall_progress"] == 0.0
    assert data["page_dot"] == 0
    assert data["all_completed"] is False


def test_api_ui_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Dubber" in response.text
    assert "Перетащите" in response.text


def test_api_setup_status():
    response = client.get("/api/setup/status")
    assert response.status_code == 200
    data = response.json()
    assert "hf_token_set" in data
    assert "models_ready" in data
    assert "cached_gb" in data


def test_api_models_download_status():
    response = client.get("/api/setup/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_downloading" in data
    assert "progress" in data


def test_api_cancel_when_idle():
    response = client.post("/api/dub/cancel")
    assert response.status_code == 400


def test_api_start_nonexistent_video():
    response = client.post(
        "/api/dub/start",
        json={"video_path": "/nonexistent/video.mp4"},
    )
    assert response.status_code == 404

