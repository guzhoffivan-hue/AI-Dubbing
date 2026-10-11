from app.core.json_store import JsonStore


def test_json_store(tmp_path):
    store = JsonStore()

    path = tmp_path / "data" / "test.json"

    data = {
        "language": "ru",
        "speakers": 3,
        "segments": [
            {
                "speaker": "SPEAKER_01",
                "text": "Привет",
            }
        ],
    }

    store.save(path, data)

    assert path.exists()
    assert store.exists(path)

    loaded = store.load(path)

    assert loaded == data