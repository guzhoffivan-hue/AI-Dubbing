"""Adapters: the only place where the core touches the engine.

Each adapter runs one engine script as a subprocess, inside the engine's own Python
environment, with the ``AI_DUB_*`` environment that ``scripts/common.py`` reads. Nothing
is imported from the engine, so the core stays dependency-free (standard library only)
and a stage can be replaced without touching the pipeline.

The stage table below mirrors ``scripts/cli.py`` (its ``phases`` list, ``outputs`` map and
``python_for``): same scripts, same virtualenvs, same extra arguments, same output files.
If ``cli.py`` changes, this table must change with it.

Safety carried over into the core:

* ``artifact`` and ``inputs`` -- the manifest fingerprints the produced file and every
  input it was derived from (including the source video), so a saved completion cannot be
  reused after its inputs change;
* ``precondition`` -- ``use_original_audio`` must be false (the same guard as
  ``cli.check_config``), and the cross-stage invariants run *before* the stage starts, so a
  stale translation can never reach TTS or the mixer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.stage_handler import StageHandler, StageOutcome
from app.invariants import validate_translation, validate_tts_timeline


class AdapterError(RuntimeError):
    """An engine stage could not be prepared or run."""


@dataclass(frozen=True)
class EngineStage:
    name: str  # the phase key used by scripts/cli.py
    script: str  # file under scripts/
    venv: str  # "asr", "translate" or "tts" (tts is resolved from config, see venv_for)
    artifact: str | None  # file produced in the work dir; None = the final video (mux)
    inputs: tuple[str, ...]  # "@input" = the source video, anything else is in the work dir
    enum_value: str  # the PipelineStage value this engine stage is registered under
    args: tuple[str, ...] = ()


# Same order and contents as the ``phases`` list in scripts/cli.py.
ENGINE_STAGES: tuple[EngineStage, ...] = (
    EngineStage("extract_audio", "extract_audio.py", "asr", "audio.wav", ("@input",), "extract_audio"),
    EngineStage("asr", "asr.py", "asr", "asr.json", ("audio.wav",), "transcription"),
    EngineStage("alignment", "align.py", "asr", "alignment.json", ("audio.wav", "asr.json"), "alignment"),
    EngineStage("diarization", "diarize.py", "asr", "diarization.json", ("audio.wav",), "diarization"),
    EngineStage(
        "segment_merge", "merge_segments.py", "asr", "segments.json",
        ("alignment.json", "diarization.json"), "segment_merge",
    ),
    EngineStage("translation", "translate.py", "translate", "translated_segments.json", ("segments.json",), "translation"),
    EngineStage(
        "dialogue", "dialogue.py", "asr", "dialogue_segments.json",
        ("segments.json", "translated_segments.json"), "dialogue",
    ),
    EngineStage("tts", "tts.py", "tts", "tts_manifest.json", ("dialogue.json",), "tts"),
    EngineStage("sfx", "sfx.py", "asr", "sfx_timeline.json", ("translated_segments.json",), "sfx", ("timeline",)),
    EngineStage("ambience", "ambience.py", "asr", "ambience.json", ("translated_segments.json",), "ambience"),
    EngineStage(
        "mix", "mix_audio.py", "asr", "mix.wav",
        ("audio.wav", "dialogue.json", "tts_manifest.json", "sfx_timeline.json", "ambience.json"),
        "audio_mix",
    ),
    EngineStage("mux", "mux.py", "asr", None, ("@input", "mix.wav"), "video_render"),
)

STAGES: dict[str, EngineStage] = {stage.name: stage for stage in ENGINE_STAGES}

# What cli.py writes instead of running the script when a stage is switched off in config.yaml.
_STUBS = {
    "sfx": ("enable_sfx", {"events": []}),
    "ambience": ("enable_ambience", {"enabled": False}),
}

# A runner performs the actual call; injectable so tests never launch ffmpeg or a model.
# Arguments: command, environment, working directory, log file.
Runner = Callable[[list, dict, Path, Path], None]


def probe_duration(video: Path) -> float:
    """Duration of a media file in seconds (same ffprobe call as cli.probe_duration)."""

    try:
        raw = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(video),
            ],
            text=True,
        )
        return float(raw.strip())
    except Exception as error:
        raise AdapterError(f"cannot read media duration of {video.name}") from error


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class EngineLayout:
    """Where the engine lives and where this project's files go."""

    root: Path  # engine root: contains scripts/, .venvs/, config.yaml
    work_dir: Path
    input_video: Path
    output_dir: Path | None = None  # default: <root>/output
    output_file: Path | None = None  # default: <output_dir>/<video stem>_ru.mp4
    start: float = 0.0  # seconds from the beginning of the source
    duration: float | None = None  # None = until the end of the source

    # -- configuration (top-level scalars of config.yaml; the core has no YAML dependency) --

    def config_value(self, key: str) -> str | None:
        path = self.root / "config.yaml"

        if not path.exists():
            return None

        match = re.search(
            rf"^{re.escape(key)}[ \t]*:[ \t]*(.*?)[ \t]*(?:#.*)?$",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )

        if match is None:
            return None

        value = match.group(1).strip().strip("'\"")

        return value or None

    def config_flag(self, key: str, default: bool | None) -> bool | None:
        value = self.config_value(key)

        if value is None:
            return default

        lowered = value.lower()

        if lowered in ("true", "yes", "on"):
            return True

        if lowered in ("false", "no", "off"):
            return False

        return default

    def check_config(self) -> None:
        """Same guard as cli.check_config: original audio must not leak into the result."""

        if self.config_flag("use_original_audio", None) is not False:
            raise AdapterError(
                "use_original_audio must be exactly false in config.yaml"
            )

    # -- paths ---------------------------------------------------------------------------

    @property
    def final_file(self) -> Path:
        directory = self.output_dir or (self.root / "output")

        return self.output_file or (directory / f"{self.input_video.stem}_ru.mp4")

    def path_of(self, name: str) -> Path:
        """Resolve an input name: '@input' is the source video, anything else the work dir."""

        if name == "@input":
            return self.input_video

        return self.work_dir / name

    def artifact_of(self, stage: str) -> Path:
        spec = self._spec(stage)

        if spec.artifact is None:
            return self.final_file

        return self.work_dir / spec.artifact

    def log_path(self, stage: str) -> Path:
        return self.root / "logs" / f"{self.work_dir.name}_{stage}.log"

    # -- environments and commands -----------------------------------------------------

    @staticmethod
    def _spec(stage: str) -> EngineStage:
        try:
            return STAGES[stage]
        except KeyError as error:
            raise AdapterError(f"unknown engine stage {stage!r}") from error

    def venv_for(self, stage: str) -> str:
        spec = self._spec(stage)

        if stage == "tts":
            backend = self.config_value("tts_backend") or "qwen3"

            return "translate" if backend == "qwen3_mlx" else "tts"

        return spec.venv

    def interpreter(self, venv: str) -> Path:
        path = self.root / ".venvs" / venv / "bin" / "python"

        if not path.exists():
            raise AdapterError(f"Python environment {venv!r} is missing ({path}); run ./setup")

        return path

    def script(self, stage: str) -> Path:
        path = self.root / "scripts" / self._spec(stage).script

        if not path.exists():
            raise AdapterError(f"missing engine script {path}")

        return path

    def resolved_duration(self) -> float:
        total = probe_duration(self.input_video)
        length = max(0.0, total - self.start)

        if self.duration is not None:
            length = min(self.duration, length)

        if length <= 0:
            raise AdapterError("the selected start is past the end of the video")

        return length

    def args(self, stage: str) -> list:
        if stage == "extract_audio":
            return [
                "--duration", str(self.resolved_duration()),
                "--start", str(self.start),
                "--output", str(self.work_dir / "audio.wav"),
            ]

        return list(self._spec(stage).args)

    def command(self, stage: str) -> list:
        return [
            str(self.interpreter(self.venv_for(stage))),
            str(self.script(stage)),
            *self.args(stage),
        ]

    def env(self, stage: str) -> dict:
        environment = dict(os.environ)
        environment["AI_DUB_WORKDIR"] = str(self.work_dir)
        environment["AI_DUB_INPUT_VIDEO"] = str(self.input_video)
        environment["AI_DUB_OUTPUT_DIR"] = str(self.output_dir or (self.root / "output"))

        if stage == "mux":  # cli.py passes the output file to mux only
            environment["AI_DUB_OUTPUT_FILE"] = str(self.final_file)

        return environment

    # -- work directory bookkeeping --------------------------------------------------------

    def write_audio_meta(self) -> float:
        """Write audio_meta.json exactly as cli.py does before the first stage."""

        length = self.resolved_duration()
        _write_json_atomic(
            self.work_dir / "audio_meta.json",
            {"duration": length, "start": self.start, "source": str(self.input_video.resolve())},
        )

        return length

    def audio_seconds(self) -> float | None:
        """Duration of the extracted audio, when it is known."""

        meta = self.work_dir / "audio_meta.json"

        if not meta.exists():
            return None

        try:
            value = float(json.loads(meta.read_text(encoding="utf-8")).get("duration"))
        except (OSError, ValueError, TypeError):
            return None

        return value if value > 0 else None

    def stub_for(self, stage: str):
        """The placeholder cli.py writes when this stage is disabled, else None."""

        rule = _STUBS.get(stage)

        if rule is None:
            return None

        flag, payload = rule

        return None if self.config_flag(flag, True) else payload

    def parameters_for(self, stage: str) -> dict:
        """Settings that change what a stage produces; for the artifact manifest's parameter hash."""

        keys = {
            "diarization": ("diarization_model", "diarization_min_speakers", "diarization_max_speakers"),
            "translation": ("translation_model",),
            "tts": ("tts_backend",),
            "sfx": ("enable_sfx",),
            "ambience": ("enable_ambience",),
        }.get(stage, ())
        parameters = {key: self.config_value(key) for key in keys}

        if stage == "extract_audio":
            parameters.update(start=self.start, duration=self.duration)

        return parameters


def subprocess_runner(command: list, env: dict, cwd: Path, log_path: Path) -> None:
    """Run a stage like cli.run_script: log everything, fail with the tail of the output."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # a sleeping Mac pauses long stages and breaks wall-clock budgets (seen on the 1200 s run)
    prefix = ["caffeinate", "-i"] if shutil.which("caffeinate") else []

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            [*prefix, *command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = completed.stdout or ""
        log.write(output)
        log.flush()

    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-12:])

        raise AdapterError(
            f"{Path(command[1]).name} failed with exit code {completed.returncode}. "
            f"Details are in {log_path}.\n{tail}"
        )


def preconditions_for(stage: str, layout: EngineLayout):
    """Checks that must pass before this stage may run."""

    def check(_context) -> None:
        layout.check_config()

        if stage in ("dialogue", "tts"):
            validate_translation(layout.work_dir)
        elif stage == "mix":
            validate_tts_timeline(layout.work_dir, audio_seconds=layout.audio_seconds())

    return check


def make_handler(
    stage: str,
    layout: EngineLayout,
    runner: Runner | None = None,
    precondition=None,
) -> StageHandler:
    """Build the pipeline handler for one engine stage."""

    spec = EngineLayout._spec(stage)
    perform = runner or subprocess_runner
    artifact = layout.artifact_of(stage)
    check = precondition if precondition is not None else preconditions_for(stage, layout)

    def run(_context) -> StageOutcome:
        stub = layout.stub_for(stage)

        if stub is not None:
            _write_json_atomic(artifact, stub)
            command: list = []
        else:
            if stage == "extract_audio":
                layout.write_audio_meta()

            command = layout.command(stage)
            perform(command, layout.env(stage), layout.root, layout.log_path(stage))

        return StageOutcome(artifact=artifact, value={"stage": stage, "command": command})

    return StageHandler(
        run=run,
        artifact=lambda _context: artifact,
        inputs=lambda _context: [layout.path_of(name) for name in spec.inputs],
        precondition=check,
    )


def make_handlers(layout: EngineLayout, runner: Runner | None = None) -> dict:
    """Handlers for every engine stage. Fails loudly if the core cannot represent one."""

    from app.models.pipeline_stage import PipelineStage

    handlers = {}
    missing = []

    for spec in ENGINE_STAGES:
        try:
            member = PipelineStage(spec.enum_value)
        except ValueError:
            missing.append(spec.enum_value)
            continue

        handlers[member] = make_handler(spec.name, layout, runner=runner)

    if missing:
        raise AdapterError(
            "PipelineStage has no member for: " + ", ".join(missing)
            + "; add them to the enum and to the stage dependency graph"
        )

    return handlers
