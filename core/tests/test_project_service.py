from app.services.project_service import ProjectService


def test_create_project(tmp_path):
    projects_dir = tmp_path / "projects"
    source_video = tmp_path / "dorama.mp4"

    source_video.write_bytes(b"fake video")

    service = ProjectService(projects_dir)
    project = service.create_project(source_video)

    assert project.name == "dorama"
    assert project.project_dir.exists()

    copied_video = project.project_dir / "source" / "dorama.mp4"

    assert copied_video.exists()
    assert copied_video.read_bytes() == b"fake video"

    assert (project.project_dir / "analysis").is_dir()
    assert (project.project_dir / "transcription").is_dir()
    assert (project.project_dir / "translation").is_dir()
    assert (project.project_dir / "voices").is_dir()
    assert (project.project_dir / "audio").is_dir()
    assert (project.project_dir / "output").is_dir()

    assert (project.project_dir / "config.json").exists()