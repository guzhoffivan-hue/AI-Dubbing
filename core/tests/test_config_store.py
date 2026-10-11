from app.core.config_store import ConfigStore
from app.models.project_config import ProjectConfig


def test_config_save_and_load(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    store = ConfigStore(project_dir)

    config = ProjectConfig.default()
    config.translation.naturalness = 0.9
    config.translation.target_language = "ru"
    config.dubbing.voice_volume = 1.2

    store.save(config)

    assert (project_dir / "config.json").exists()

    loaded_config = store.load()

    assert loaded_config.translation.naturalness == 0.9
    assert loaded_config.translation.target_language == "ru"
    assert loaded_config.dubbing.voice_volume == 1.2