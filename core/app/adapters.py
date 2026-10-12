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
  stale translation can never reach TTS or the mixer;
* ``cancel`` -- the stage subprocess is polled and stopped on a user cancel or a
  stage timeout, which ``cli.run_script`` cannot do because it blocks in ``subprocess.run``.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.cancellation import CancelToken
from app.core.progress import LineCallback, LogProgress
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
    EngineStage("tts", "tts.py", "tts", "tts_manifest.json", ("dialogue.json", "speakers.json"), "tts"),
    EngineStage("sfx", "sfx.py", "asr", "sfx_timeline.json", ("translation.json", "@sfx_index"), "sfx", ("timeline",)),
    EngineStage("ambience", "ambience.py", "asr", "ambience.json", ("translation.json", "@sfx_index"), "ambience"),
    EngineStage(
        "mix", "mix_audio.py", "asr", "mix.wav",
        ("audio_meta.json", "tts_manifest.json", "sfx_timeline.json", "ambience.json"),
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
class Runner(Protocol):
    """Executes one stage command.

    ``cancel`` is a cooperative token and ``on_line`` receives the stage's output as
    it arrives. A runner that owns a subprocess must poll the token and stream the
    lines: that is what lets a user cancel or a stage timeout stop the work, and what
    lets a long stage report progress instead of looking hung
    (see app.core.cancellation and app.core.progress).
    """

    def __call__(
        self,
        command: list,
        env: dict,
        cwd: Path,
        log_path: Path,
        cancel: CancelToken | None = None,
        on_line: LineCallback | None = None,
    ) -> None: ...


#: How often a running stage checks the cancel token.
POLL_SECONDS = 0.5


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
        """Resolve an input name: '@input' is the source video, '@sfx_index' the
        library-wide index (sfx.py and ambience.py read it from ROOT/work),
        anything else a file of this project's work dir."""

        if name == "@input":
            return self.input_video

        if name == "@sfx_index":
            return self.root / "work" / "sfx_index.json"

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


def _stop_process(process: subprocess.Popen) -> None:
    """Stop a stage process and everything it spawned.

    The process is started in its own session, so the signal reaches the group:
    ``caffeinate`` is only the parent of the real stage, and terminating it alone
    would leave the stage running.
    """

    if process.poll() is not None:
        return

    for stop_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), stop_signal)
        except (ProcessLookupError, PermissionError):
            process.kill()

        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def subprocess_runner(
    command: list,
    env: dict,
    cwd: Path,
    log_path: Path,
    cancel: CancelToken | None = None,
    on_line: LineCallback | None = None,
) -> None:
    """Run a stage like cli.run_script: log everything, fail with the tail of the output.

    Unlike ``cli.run_script`` this streams the output instead of blocking in
    ``subprocess.run``, so a cancelled or timed out stage stops its own process and
    a long stage can report progress while it works.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # a sleeping Mac pauses long stages and breaks wall-clock budgets (seen on the 1200 s run)
    prefix = ["caffeinate", "-i"] if shutil.which("caffeinate") else []

    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(command) + "\n")
        log.flush()

        process = subprocess.Popen(
            [*prefix, *command],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        lines: queue.Queue = queue.Queue()

        def pump() -> None:
            # Reading in a thread keeps the main loop free to watch the cancel token.
            stream = process.stdout

            try:
                if stream is not None:
                    for line in stream:
                        lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()

        collected: list[str] = []

        while True:
            try:
                line = lines.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if cancel is not None and cancel.cancelled:
                    _stop_process(process)

                    while True:
                        try:
                            leftover = lines.get(timeout=POLL_SECONDS)
                        except queue.Empty:
                            break

                        if leftover is None:
                            break

                        collected.append(leftover)
                        log.write(leftover)

                    log.write(f"\n[core] stage stopped ({cancel.reason})\n")
                    log.flush()

                    # Raises StageCancelled or StageTimeout, so the pipeline records
                    # why the stage stopped instead of blaming the exit code.
                    cancel.raise_if_cancelled()

                continue

            if line is None:
                break

            collected.append(line)
            log.write(line)
            log.flush()

            if on_line is not None:
                on_line(line)

        process.wait()
        reader.join(timeout=5)

    output = "".join(collected)

    if process.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-12:])

        raise AdapterError(
            f"{Path(command[1]).name} failed with exit code {process.returncode}. "
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


def _json_line_count(path: Path) -> int | None:
    """How many speech lines a stage is about to process, when the input is readable."""

    try:
        return len(json.loads(path.read_text(encoding="utf-8"))["lines"])
    except (OSError, ValueError, TypeError, KeyError):
        return None


def progress_for(stage: str, layout: EngineLayout) -> LogProgress | None:
    """How this stage's own log says how far it has got.

    Only the long stages get a reader: the short ones finish in seconds, and a fake
    bar would be worse than none. Diarization prints no per-unit line of its own, so
    it honestly reports no intra-stage progress.
    """

    if stage == "asr":
        return LogProgress(
            (r"ASR Chunk (\d+)/(\d+) completed",),
            total_pattern=r"Splitting into (\d+) safe chunks",
        )

    if stage == "alignment":
        return LogProgress((r"Aligned chunk (\d+)/(\d+)",))

    if stage == "translation":
        return LogProgress(
            (
                r"Translated (\d+)/(\d+) LINE_",
                r"\[(\d+)/(\d+)\] Translating LINE_",
            )
        )

    if stage == "tts":
        return LogProgress(
            (
                r"segment (\d+) speaker=",
                r"Reused synthesized line (\d+)/(\d+)",
            ),
            total=_json_line_count(layout.work_dir / "dialogue.json"),
        )

    return None


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
            perform(
                command,
                layout.env(stage),
                layout.root,
                layout.log_path(stage),
                cancel=getattr(_context, "cancel", None),
                on_line=getattr(_context, "on_output_line", None),
            )

        return StageOutcome(artifact=artifact, value={"stage": stage, "command": command})

    return StageHandler(
        run=run,
        artifact=lambda _context: artifact,
        inputs=lambda _context: [layout.path_of(name) for name in spec.inputs],
        parameters=lambda _context: layout.parameters_for(stage),
        precondition=check,
        progress=progress_for(stage, layout),
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
