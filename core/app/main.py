"""FastAPI application providing HTTP and SSE API for the AI-Dubbing pipeline.

Serves as the backend for the native macOS AI Dubber application and CLI tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.adapters import ENGINE_STAGES, EngineLayout, make_handlers
from app.core.cancellation import CancelToken
from app.core.pipeline import Pipeline, PipelineProgress
from app.core.pipeline_context import PipelineContext
from app.core.pipeline_state import PipelineState
from app.core.stage_dependencies import STAGE_DEPENDENCIES
from app.models.pipeline_stage import PipelineStage
from app.models.project_config import ProjectConfig
from app.models.stage_state import StageStatus
from projects.project import Project

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = Path(__file__).resolve().parent / "ui"

REQUIRED_HF_REPOS = [
    "Qwen/Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ForcedAligner-0.6B",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
]

STAGE_TO_DOT: dict[str, int] = {
    PipelineStage.EXTRACT_AUDIO.value: 1,
    PipelineStage.TRANSCRIPTION.value: 1,
    PipelineStage.ALIGNMENT.value: 1,
    PipelineStage.DIARIZATION.value: 2,
    PipelineStage.SEGMENT_MERGE.value: 2,
    PipelineStage.TRANSLATION.value: 3,
    PipelineStage.DIALOGUE.value: 3,
    PipelineStage.TTS.value: 4,
    PipelineStage.SFX.value: 4,
    PipelineStage.AMBIENCE.value: 4,
    PipelineStage.AUDIO_MIX.value: 5,
    PipelineStage.VIDEO_RENDER.value: 5,
}

app = FastAPI(
    title="AI-Dubbing Core API",
    description="Offline video dubbing pipeline API for Apple Silicon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- State management for active dubbing job ---------------------------------

class ActiveJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.pipeline: Pipeline | None = None
        self.cancel_token: CancelToken | None = None
        self.layout: EngineLayout | None = None
        self.is_running: bool = False
        self.started_at: float = 0.0
        self.video_path: Path | None = None
        self.recent_events: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def status_summary(self) -> dict[str, Any]:
        with self.lock:
            if not self.pipeline or not self.pipeline.state:
                return {
                    "is_running": False,
                    "video": str(self.video_path) if self.video_path else None,
                    "stages": {},
                    "overall_progress": 0.0,
                    "current_stage": None,
                    "page_dot": 0,
                    "all_completed": False,
                    "output_video": None,
                    "started_at": None,
                    "elapsed": 0.0,
                    "last_error": self.last_error,
                }

            stages_dict = {}
            all_stages = list(PipelineStage)
            total_count = len(all_stages)
            completed_units = 0.0
            completed_count = 0
            current_stage: str | None = None
            last_active_stage: str | None = None
            failed_message: str | None = None

            for stage in all_stages:
                st = self.pipeline.state.get(stage.value)
                stages_dict[stage.value] = {
                    "status": st.status.value,
                    "progress": st.progress,
                    "message": st.message,
                }
                if st.status == StageStatus.COMPLETED:
                    completed_units += 1.0
                    completed_count += 1
                    last_active_stage = stage.value
                elif st.status == StageStatus.RUNNING:
                    completed_units += max(0.0, min(1.0, float(st.progress)))
                    current_stage = stage.value
                elif st.status == StageStatus.FAILED and not failed_message:
                    failed_message = f"{stage.value}: {st.message}"

            all_completed = completed_count == total_count and total_count > 0
            active_key = current_stage or last_active_stage
            if all_completed:
                page_dot = 5
                overall_progress = 1.0
            elif active_key:
                page_dot = STAGE_TO_DOT.get(active_key, 1)
                overall_progress = round(completed_units / total_count, 4) if total_count else 0.0
            else:
                page_dot = 1 if self.is_running else 0
                overall_progress = 0.0

            output_video = str(self.layout.final_file) if self.layout else None
            elapsed = time.monotonic() - self.started_at if self.is_running else 0.0
            return {
                "is_running": self.is_running,
                "video": str(self.video_path) if self.video_path else None,
                "stages": stages_dict,
                "overall_progress": overall_progress,
                "current_stage": active_key,
                "page_dot": page_dot,
                "all_completed": all_completed,
                "output_video": output_video,
                "started_at": self.started_at if self.is_running else None,
                "elapsed": round(elapsed, 1),
                "last_error": self.last_error or failed_message,
            }


JOB = ActiveJob()


# -- State management for model downloads ------------------------------------

class ModelDownloadState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_downloading: bool = False
        self.paused: bool = False
        self.cancelled: bool = False
        self.progress: float = 0.0
        self.speed_mbps: float = 25.0
        self.eta_min: int = 3
        self.caption: str = ""
        self.completed: bool = False
        self.total_gb: float = 8.5


MODELS_STATE = ModelDownloadState()


def _read_hf_token() -> str | None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line_s = line.strip()
            if line_s.startswith("HF_TOKEN="):
                val = line_s.split("=", 1)[1].strip().strip('"').strip("'")
                if len(val) > 5:
                    return val
    return None


def _check_models_cache() -> tuple[bool, list[str], float]:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    missing: list[str] = []
    for repo in REQUIRED_HF_REPOS:
        folder_name = "models--" + repo.replace("/", "--")
        repo_path = hub_dir / folder_name
        snapshots_dir = repo_path / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            missing.append(repo)
    return (len(missing) == 0, missing, 8.5)


# -- Request Models ----------------------------------------------------------

class DubStartRequest(BaseModel):
    video_path: str = Field(..., description="Path to input video file")
    duration: float | None = Field(None, description="Optional clip duration in seconds")
    start: float = Field(0.0, description="Start offset in seconds")
    resume: bool = Field(True, description="Resume completed stages")


class ConfigUpdateRequest(BaseModel):
    config: dict[str, Any]


class TokenSaveRequest(BaseModel):
    token: str


class RevealRequest(BaseModel):
    path: str = ""


# -- UI Endpoint -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui() -> FileResponse:
    """Serve the native macOS UI frontend."""
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    return FileResponse(index_path, media_type="text/html")


# -- Setup & System Endpoints ------------------------------------------------

@app.get("/api/setup/status")
def get_setup_status() -> dict[str, Any]:
    """Check HF token and cached models readiness for first-launch alerts."""
    token = _read_hf_token()
    models_ready, missing, cached_gb = _check_models_cache()
    return {
        "hf_token_set": token is not None,
        "models_ready": models_ready,
        "missing_models": missing,
        "cached_gb": cached_gb,
    }


@app.post("/api/setup/token")
def save_setup_token(req: TokenSaveRequest) -> dict[str, Any]:
    """Save HF_TOKEN to .env in project root."""
    token = req.token.strip()
    if not token:
        raise HTTPException(status_code=400, detail="Empty token")

    env_file = ROOT / ".env"
    lines: list[str] = []
    replaced = False
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                lines.append(f"HF_TOKEN={token}")
                replaced = True
            else:
                lines.append(line)
    if not replaced:
        lines.append(f"HF_TOKEN={token}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["HF_TOKEN"] = token
    return {"status": "saved", "hf_token_set": True}


@app.post("/api/setup/models/download")
def start_models_download() -> dict[str, Any]:
    """Start or verify required Hugging Face models in background."""
    with MODELS_STATE.lock:
        if MODELS_STATE.is_downloading:
            return {"status": "already_downloading"}
        MODELS_STATE.is_downloading = True
        MODELS_STATE.paused = False
        MODELS_STATE.cancelled = False
        MODELS_STATE.completed = False
        MODELS_STATE.progress = 0.05
        MODELS_STATE.caption = "Проверка и загрузка весов моделей…"

    def _download_worker() -> None:
        try:
            from huggingface_hub import snapshot_download
            token = _read_hf_token()
            total = len(REQUIRED_HF_REPOS)
            for idx, repo in enumerate(REQUIRED_HF_REPOS):
                while True:
                    with MODELS_STATE.lock:
                        if MODELS_STATE.cancelled:
                            MODELS_STATE.is_downloading = False
                            return
                        is_paused = MODELS_STATE.paused
                    if not is_paused:
                        break
                    time.sleep(0.3)

                short_name = repo.split("/")[-1]
                with MODELS_STATE.lock:
                    MODELS_STATE.progress = round((idx + 0.2) / total, 2)
                    MODELS_STATE.caption = f"Загрузка {short_name} ({idx + 1}/{total})…"

                folder_name = "models--" + repo.replace("/", "--")
                snapshots_dir = Path.home() / ".cache" / "huggingface" / "hub" / folder_name / "snapshots"
                if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                    time.sleep(0.35)
                else:
                    snapshot_download(repo_id=repo, token=token)

                with MODELS_STATE.lock:
                    MODELS_STATE.progress = round((idx + 1) / total, 2)

            with MODELS_STATE.lock:
                MODELS_STATE.progress = 1.0
                MODELS_STATE.completed = True
                MODELS_STATE.is_downloading = False
                MODELS_STATE.caption = "Все модели готовы к работе"
        except Exception as exc:
            with MODELS_STATE.lock:
                MODELS_STATE.caption = f"Ошибка загрузки: {exc}"
                MODELS_STATE.is_downloading = False

    threading.Thread(target=_download_worker, daemon=True).start()
    return {"status": "started"}


@app.post("/api/setup/models/pause")
def pause_models_download() -> dict[str, Any]:
    with MODELS_STATE.lock:
        MODELS_STATE.paused = not MODELS_STATE.paused
        paused = MODELS_STATE.paused
        if paused:
            MODELS_STATE.caption = "Загрузка приостановлена"
    return {"status": "paused" if paused else "resumed", "paused": paused}


@app.post("/api/setup/models/cancel")
def cancel_models_download() -> dict[str, Any]:
    with MODELS_STATE.lock:
        MODELS_STATE.cancelled = True
        MODELS_STATE.is_downloading = False
    return {"status": "cancelled"}


@app.get("/api/setup/models/status")
def get_models_download_status() -> dict[str, Any]:
    with MODELS_STATE.lock:
        return {
            "is_downloading": MODELS_STATE.is_downloading,
            "paused": MODELS_STATE.paused,
            "progress": MODELS_STATE.progress,
            "speed_mbps": MODELS_STATE.speed_mbps,
            "eta_min": MODELS_STATE.eta_min,
            "caption": MODELS_STATE.caption,
            "completed": MODELS_STATE.completed,
            "total_gb": MODELS_STATE.total_gb,
        }


@app.post("/api/system/reveal")
def reveal_in_finder(req: RevealRequest) -> dict[str, Any]:
    """Reveal output file or output folder in macOS Finder."""
    target = Path(req.path) if req.path else (ROOT / "output")
    if not target.is_absolute():
        target = ROOT / target
    if target.exists():
        if target.is_file():
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["open", str(target)])
        return {"status": "opened", "path": str(target)}
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    subprocess.Popen(["open", str(out_dir)])
    return {"status": "opened_fallback", "path": str(out_dir)}


# -- Core Pipeline Endpoints -------------------------------------------------

@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """System health check: ffmpeg, venvs, models cache, sfx library."""
    ffmpeg_ok = shutil.which("ffmpeg") is not None or Path("/opt/homebrew/bin/ffmpeg").exists()
    ffprobe_ok = shutil.which("ffprobe") is not None or Path("/opt/homebrew/bin/ffprobe").exists()

    venvs = {
        name: (ROOT / ".venvs" / name / "bin" / "python").exists()
        for name in ("asr", "translate", "tts", "diarize")
    }

    sfx_wavs = list((ROOT / "sfx_library").rglob("*.wav")) if (ROOT / "sfx_library").exists() else []
    hf_token_set = _read_hf_token() is not None

    return {
        "status": "ok",
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "venvs": venvs,
        "sfx_library": {
            "exists": (ROOT / "sfx_library").exists(),
            "wav_count": len(sfx_wavs),
        },
        "hf_token_set": hf_token_set,
        "platform": "darwin",
    }


@app.get("/api/stages")
def get_stages() -> list[dict[str, Any]]:
    """Return all 12 pipeline stages in dependency order with their declared specs."""
    result = []
    for spec in ENGINE_STAGES:
        stage_enum = PipelineStage(spec.enum_value)
        deps = [d.value for d in STAGE_DEPENDENCIES.get(stage_enum, [])]
        result.append({
            "key": spec.name,
            "enum_value": spec.enum_value,
            "script": spec.script,
            "venv": spec.venv,
            "artifact": spec.artifact,
            "dependencies": deps,
        })
    return result


@app.get("/api/videos")
def list_videos() -> dict[str, Any]:
    """List available input videos and rendered output dubs."""
    input_dir = ROOT / "input"
    output_dir = ROOT / "output"

    inputs = []
    if input_dir.exists():
        for ext in ("*.mp4", "*.mov", "*.mkv", "*.m4v"):
            for p in input_dir.glob(ext):
                stat = p.stat()
                inputs.append({
                    "path": str(p),
                    "name": p.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                })

    outputs = []
    if output_dir.exists():
        for p in output_dir.glob("*_ru.mp4"):
            stat = p.stat()
            outputs.append({
                "path": str(p),
                "name": p.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })

    return {"input": inputs, "output": outputs}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Read current config.yaml."""
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest) -> dict[str, Any]:
    """Update config.yaml safely."""
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(req.config, f, allow_unicode=True)
    return {"status": "updated"}


@app.get("/api/dub/status")
def get_dub_status() -> dict[str, Any]:
    """Get active or recent dubbing progress and stage states."""
    return JOB.status_summary()


@app.post("/api/dub/cancel")
def cancel_dub() -> dict[str, Any]:
    """Cancel currently running dubbing job."""
    with JOB.lock:
        if not JOB.is_running or not JOB.cancel_token:
            raise HTTPException(status_code=400, detail="No active job running")
        JOB.cancel_token.request("Cancelled via API")
    return {"status": "cancel_requested"}


@app.post("/api/dub/start")
def start_dub(req: DubStartRequest) -> dict[str, Any]:
    """Start dubbing on the given video."""
    with JOB.lock:
        if JOB.is_running:
            raise HTTPException(status_code=409, detail="A dubbing job is already in progress")

    video = Path(req.video_path)
    if not video.is_absolute():
        candidate = ROOT / video
        if not candidate.exists() and (ROOT / "input" / video.name).exists():
            video = ROOT / "input" / video.name
        else:
            video = candidate
    if not video.exists():
        raise HTTPException(status_code=404, detail=f"Source video not found: {video}")

    duration = req.duration
    slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in video.stem).strip("._") or "video"
    if duration is not None:
        work_dir = ROOT / "work" / f"{slug}_{int(duration)}s"
    else:
        work_dir = ROOT / "work" / slug
    work_dir.mkdir(parents=True, exist_ok=True)

    layout = EngineLayout(
        root=ROOT,
        work_dir=work_dir,
        input_video=video,
        output_dir=ROOT / "output",
        start=req.start,
        duration=duration,
    )
    layout.check_config()

    handlers = make_handlers(layout)
    project_dir = work_dir / "_core_state"
    project_dir.mkdir(exist_ok=True)

    state = PipelineState.load(project_dir) if (req.resume and (project_dir / "pipeline_state.json").exists()) else PipelineState()
    cancel = CancelToken()

    context = PipelineContext(
        project=Project(
            id=uuid4(),
            name=video.stem,
            source_video=video,
            project_dir=project_dir,
            config=ProjectConfig.default(),
        )
    )

    def on_progress(p: PipelineProgress) -> None:
        pass

    pipe = Pipeline(
        state=state,
        context=context,
        handlers=handlers,
        on_progress=on_progress,
        cancel=cancel,
        project_dir=project_dir,
    )

    def worker_run() -> None:
        try:
            pipe.run(context)
        except Exception as exc:
            JOB.last_error = str(exc)
        finally:
            with JOB.lock:
                JOB.is_running = False

    t = threading.Thread(target=worker_run, daemon=True)
    with JOB.lock:
        JOB.pipeline = pipe
        JOB.cancel_token = cancel
        JOB.layout = layout
        JOB.is_running = True
        JOB.started_at = time.monotonic()
        JOB.video_path = video
        JOB.thread = t
        JOB.last_error = None

    t.start()
    return {"status": "started", "work_dir": str(work_dir)}


@app.get("/api/dub/events")
async def sse_events() -> StreamingResponse:
    """Server-Sent Events endpoint streaming pipeline progress in real-time."""
    async def event_generator() -> AsyncGenerator[str, None]:
        last_summary = None
        while True:
            summary = JOB.status_summary()
            if summary != last_summary:
                last_summary = summary
                yield f"data: {json.dumps(summary, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/logs/{stage}")
def get_stage_log(stage: str, lines: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Retrieve the log tail for a given pipeline stage."""
    with JOB.lock:
        layout = JOB.layout

    if layout is None:
        raise HTTPException(status_code=404, detail="No active or recent layout found")

    try:
        log_file = layout.log_path(stage)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not log_file.exists():
        return {"stage": stage, "log": "", "exists": False}

    content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(content[-lines:])
    return {"stage": stage, "log": tail, "total_lines": len(content), "exists": True}
