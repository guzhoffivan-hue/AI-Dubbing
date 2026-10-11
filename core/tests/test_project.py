from pathlib import Path

from projects.project_manager import ProjectManager


def test_create_project(tmp_path):
    projects_dir = tmp_path / "projects"
    source_video = tmp_path / "dorama.mp4"

    source_video.touch()

    manager = ProjectManager(projects_dir)
    project = manager.create_project(source_video)

    assert project.name == "dorama"
    assert project.source_video == source_video
    assert project.project_dir.exists()

    expected_directories = [
        "source",
        "analysis",
        "transcription",
        "translation",
        "voices",
        "audio",
        "output",
    ]

    for directory in expected_directories:
        assert (project.project_dir / directory).is_dir()