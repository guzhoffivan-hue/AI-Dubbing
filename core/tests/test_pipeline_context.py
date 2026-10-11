from pathlib import Path

from app.core.pipeline_context import PipelineContext
from projects.project import Project


def test_pipeline_context(tmp_path):
    source_video = tmp_path / "dorama.mp4"
    source_video.touch()

    projects_dir = tmp_path / "projects"

    project = Project.create(
        source_video=source_video,
        projects_dir=projects_dir,
    )

    project.project_dir.mkdir(parents=True)

    context = PipelineContext(project=project)

    assert context.source_video == source_video
    assert context.project_dir == project.project_dir

    assert context.analysis_dir == project.project_dir / "analysis"
    assert context.transcription_dir == project.project_dir / "transcription"
    assert context.translation_dir == project.project_dir / "translation"
    assert context.voices_dir == project.project_dir / "voices"
    assert context.audio_dir == project.project_dir / "audio"
    assert context.output_dir == project.project_dir / "output"