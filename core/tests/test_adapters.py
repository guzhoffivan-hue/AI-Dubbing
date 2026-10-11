import json, tempfile, unittest
from pathlib import Path
from unittest import mock

from app import adapters, invariants
from app.adapters import AdapterError, EngineLayout, ENGINE_STAGES, make_handler, make_handlers


def build(tmp, config="use_original_audio: false\ntts_backend: qwen3_mlx\n"):
    root = Path(tmp) / "engine"
    (root / "scripts").mkdir(parents=True)
    for spec in ENGINE_STAGES:
        (root / "scripts" / spec.script).write_text("")
    for venv in ("asr", "translate", "tts"):
        py = root / ".venvs" / venv / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("")
    (root / "config.yaml").write_text(config)
    video = Path(tmp) / "video.mp4"
    video.write_bytes(b"x")
    return EngineLayout(root=root, work_dir=root / "work" / "video", input_video=video)


class Recorder:
    def __init__(self): self.calls = []
    def __call__(self, command, env, cwd, log): self.calls.append((command, env, cwd, log))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.addCleanup(self._tmp.cleanup)
        self.layout = build(self._tmp.name)
        invariants.CALLS.clear()

    def test_twelve_stages_match_cli_order(self):
        self.assertEqual([s.name for s in ENGINE_STAGES],
            ["extract_audio","asr","alignment","diarization","segment_merge","translation",
             "dialogue","tts","sfx","ambience","mix","mux"])

    def test_venv_selection_follows_cli(self):
        l = self.layout
        self.assertEqual(l.venv_for("asr"), "asr")
        self.assertEqual(l.venv_for("diarization"), "asr")
        self.assertEqual(l.venv_for("translation"), "translate")
        self.assertEqual(l.venv_for("tts"), "translate")          # qwen3_mlx backend
        (l.root / "config.yaml").write_text("use_original_audio: false\ntts_backend: qwen3\n")
        self.assertEqual(l.venv_for("tts"), "tts")
        (l.root / "config.yaml").write_text("use_original_audio: false\n")
        self.assertEqual(l.venv_for("tts"), "tts")                # cli default is qwen3

    def test_extract_audio_gets_duration_start_output(self):
        with mock.patch.object(adapters, "probe_duration", return_value=1000.0):
            layout = EngineLayout(**{**self.layout.__dict__, "start": 10.0, "duration": 150.0})
            cmd = layout.command("extract_audio")
        self.assertEqual(cmd[2:], ["--duration","150.0","--start","10.0","--output",str(layout.work_dir/"audio.wav")])

    def test_full_mode_uses_whole_video(self):
        with mock.patch.object(adapters, "probe_duration", return_value=240.5):
            self.assertEqual(self.layout.resolved_duration(), 240.5)

    def test_start_past_end_fails(self):
        layout = EngineLayout(**{**self.layout.__dict__, "start": 500.0})
        with mock.patch.object(adapters, "probe_duration", return_value=100.0):
            with self.assertRaises(AdapterError): layout.resolved_duration()

    def test_run_extract_audio_writes_audio_meta_then_runs(self):
        rec = Recorder()
        with mock.patch.object(adapters, "probe_duration", return_value=60.0):
            outcome = make_handler("extract_audio", self.layout, runner=rec).run(None)
        meta = json.loads((self.layout.work_dir / "audio_meta.json").read_text())
        self.assertEqual(meta["duration"], 60.0)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(outcome.artifact, self.layout.work_dir / "audio.wav")
        self.assertEqual(self.layout.audio_seconds(), 60.0)

    def test_sfx_gets_timeline_arg(self):
        self.assertEqual(self.layout.command("sfx")[2:], ["timeline"])

    def test_output_file_env_only_for_mux(self):
        self.assertNotIn("AI_DUB_OUTPUT_FILE", self.layout.env("asr"))
        self.assertTrue(self.layout.env("mux")["AI_DUB_OUTPUT_FILE"].endswith("video_ru.mp4"))
        self.assertEqual(self.layout.artifact_of("mux"), self.layout.final_file)

    def test_disabled_sfx_and_ambience_write_stubs_and_launch_nothing(self):
        (self.layout.root / "config.yaml").write_text(
            "use_original_audio: false\nenable_sfx: false   # off\nenable_ambience: false\n")
        rec = Recorder()
        make_handler("sfx", self.layout, runner=rec).run(None)
        make_handler("ambience", self.layout, runner=rec).run(None)
        self.assertEqual(rec.calls, [])
        self.assertEqual(json.loads((self.layout.work_dir/"sfx_timeline.json").read_text()), {"events": []})
        self.assertEqual(json.loads((self.layout.work_dir/"ambience.json").read_text()), {"enabled": False})

    def test_enabled_by_default_when_key_absent(self):
        self.assertIsNone(self.layout.stub_for("sfx"))

    def test_original_audio_guard(self):
        for text in ("use_original_audio: true\n", "tts_backend: qwen3\n", ""):
            (self.layout.root / "config.yaml").write_text(text)
            with self.assertRaises(AdapterError):
                make_handler("asr", self.layout, runner=Recorder()).precondition(None)
        (self.layout.root / "config.yaml").write_text("use_original_audio: false\n")
        make_handler("asr", self.layout, runner=Recorder()).precondition(None)

    def test_translation_invariant_runs_before_dialogue_and_tts(self):
        for stage in ("dialogue", "tts"):
            make_handler(stage, self.layout).precondition(None)
        self.assertEqual([c[0] for c in invariants.CALLS], ["translation", "translation"])

    def test_mix_invariant_gets_audio_seconds(self):
        self.layout.work_dir.mkdir(parents=True)
        (self.layout.work_dir/"audio_meta.json").write_text('{"duration": 1200.0}')
        make_handler("mix", self.layout).precondition(None)
        self.assertEqual(invariants.CALLS[-1], ("tts", self.layout.work_dir, 1200.0))

    def test_inputs_resolve_source_video_and_work_files(self):
        handler = make_handler("mux", self.layout)
        self.assertEqual(handler.inputs(None), [self.layout.input_video, self.layout.work_dir / "mix.wav"])

    def test_missing_script_or_venv_fails_loudly(self):
        (self.layout.root/"scripts"/"asr.py").unlink()
        with self.assertRaises(AdapterError): self.layout.command("asr")
        (self.layout.root/".venvs"/"translate"/"bin"/"python").unlink()
        with self.assertRaises(AdapterError): self.layout.command("translation")

    def test_unknown_stage(self):
        with self.assertRaises(AdapterError): make_handler("nope", self.layout)

    def test_make_handlers_registers_all_and_fails_on_missing_enum(self):
        self.assertEqual(len(make_handlers(self.layout)), 12)
        from app.models import pipeline_stage as ps
        class Short(str, ps.Enum):
            TTS = "tts"
        with mock.patch.object(ps, "PipelineStage", Short):
            with self.assertRaises(AdapterError) as ctx: make_handlers(self.layout)
        self.assertIn("extract_audio", str(ctx.exception))

    def test_subprocess_runner_logs_and_reports_tail(self):
        log = self.layout.root / "logs" / "x.log"
        py = self.layout.root/"scripts"/"boom.py"
        py.write_text("import sys\nprint('line1')\nprint('bad thing',file=sys.stderr)\nsys.exit(3)\n")
        with self.assertRaises(AdapterError) as ctx:
            adapters.subprocess_runner(["python3", str(py)], {}, self.layout.root, log)
        self.assertIn("exit code 3", str(ctx.exception)); self.assertIn("bad thing", str(ctx.exception))
        self.assertIn("line1", log.read_text())
        ok = self.layout.root/"scripts"/"ok.py"; ok.write_text("print('fine')\n")
        adapters.subprocess_runner(["python3", str(ok)], {}, self.layout.root, log)
        self.assertIn("fine", log.read_text())

    def test_parameters_cover_config_that_changes_output(self):
        (self.layout.root / "config.yaml").write_text(
            "use_original_audio: false\ntts_backend: qwen3_mlx\ntranslation_model: m/qwen\n")
        self.assertEqual(self.layout.parameters_for("tts"), {"tts_backend": "qwen3_mlx"})
        self.assertEqual(self.layout.parameters_for("translation"), {"translation_model": "m/qwen"})

if __name__ == "__main__":
    unittest.main()
