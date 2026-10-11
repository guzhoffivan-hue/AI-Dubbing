#!/usr/bin/env python3
"""Staged-run watchdog for the 1200-second validation test.

Purpose: run the *existing* orchestrator (test_root/scripts/cli.py) unmodified, but
stop it cleanly once a chosen stage is recorded as done in status.json. This lets the
20-minute test be validated in stages (extract -> ASR -> alignment -> diarization ->
segment merge) before committing hours to translation and TTS.

It does not modify the pipeline and makes no decisions about artifacts: it only starts
cli.py in its own process group, watches status.json, and terminates the whole group
when the target stage completes.

SAFETY: everything is resolved relative to this file's own directory, and guard()
refuses to start unless that directory really is the isolated test root. An earlier
version of this wrapper used parents[1], resolved to the live project, and started the
live pipeline against the wrong work directory; guard() exists so that cannot recur.

Usage: staged_run.py [--dry-run] [stop_after_stage] [duration_seconds] [max_seconds]
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# --- paths, resolved strictly relative to THIS file --------------------------
ROOT = Path(__file__).resolve().parent          # test_run_1200s/
LIVE_PROJECT = Path('/Volumes/develop/AI-Dubbing')
EXPECTED_INPUT = '42503442096_first1200s.mp4'
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.m4v', '.avi'}

DRY_RUN = '--dry-run' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
STOP_AFTER = args[0] if len(args) > 0 else 'segment_merge'
DURATION = args[1] if len(args) > 1 else '1200'
MAX_SECONDS = float(args[2]) if len(args) > 2 else 5400.0

PYTHON = ROOT / '.venvs' / 'asr' / 'bin' / 'python'
CLI = ROOT / 'scripts' / 'cli.py'
LOG = ROOT / 'logs' / 'staged_run_wrapper.log'
CLI_LOG = ROOT / 'logs' / 'cli_stdout.log'
CMD = [str(PYTHON), str(CLI), 'dub', '--test', '--duration', str(DURATION)]


def guard() -> None:
    """Refuse to run unless this really is the isolated 1200s test root."""
    problems = []
    if ROOT == LIVE_PROJECT:
        problems.append('resolved to the LIVE project root, not the test root')
    if not (ROOT / '.venvs').is_symlink():
        problems.append('.venvs is not the expected symlink into the live project')
    if not CLI.exists():
        problems.append(f'missing orchestrator: {CLI}')
    inputs = sorted(p for p in (ROOT / 'input').iterdir() if p.suffix.lower() in VIDEO_SUFFIXES) \
        if (ROOT / 'input').is_dir() else []
    if [p.name for p in inputs] != [EXPECTED_INPUT]:
        problems.append(f'input/ must contain exactly {EXPECTED_INPUT}, found {[p.name for p in inputs]}')
    statuses = list((ROOT / 'work').glob('*/status.json')) if (ROOT / 'work').is_dir() else []
    if len(statuses) > 1:
        problems.append(f'more than one run state in work/: {[p.parent.name for p in statuses]}')
    if problems:
        raise SystemExit('REFUSING TO RUN — ' + '; '.join(problems))


def stamp(message: str) -> None:
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {message}'
    print(line, flush=True)
    with LOG.open('a') as handle:
        handle.write(line + '\n')


def status_snapshot() -> dict:
    merged: dict[str, str] = {}
    for path in sorted((ROOT / 'work').glob('*/status.json')):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for key, value in data.items():
            if not key.startswith('_') and isinstance(value, str):
                merged[key] = value
    return merged


def main() -> int:
    guard()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if DRY_RUN:
        print('DRY RUN — resolved paths:')
        print(f'  ROOT        = {ROOT}')
        print(f'  CLI         = {CLI}')
        print(f'  PYTHON      = {PYTHON}')
        print(f'  work        = {ROOT / "work"}')
        print(f'  logs        = {ROOT / "logs"}')
        print(f'  output      = {ROOT / "output"}')
        print(f'  command     = {" ".join(CMD)}')
        print(f'  stop after  = {STOP_AFTER}')
        print(f'  guard       = PASSED (this is the isolated test root)')
        return 0

    stamp(f'start: {" ".join(CMD)}   (stop after stage: {STOP_AFTER})')
    proc = subprocess.Popen(CMD, cwd=str(ROOT), stdout=CLI_LOG.open('a'),
                            stderr=subprocess.STDOUT, start_new_session=True)
    stamp(f'cli.py pid={proc.pid} (new session/process group)')
    started = time.monotonic()
    previous = status_snapshot()
    stopped_for_target = False
    try:
        while True:
            returncode = proc.poll()
            if returncode is not None:
                stamp(f'cli.py exited on its own with code {returncode}')
                break
            current = status_snapshot()
            for stage, value in current.items():
                if previous.get(stage) != value:
                    stamp(f'stage {stage} -> {value}')
            previous = current
            if current.get(STOP_AFTER) == 'done':
                stamp(f'target stage {STOP_AFTER} is done; terminating process group {proc.pid}')
                stopped_for_target = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stamp('group did not exit on SIGTERM; sending SIGKILL')
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=30)
                break
            if time.monotonic() - started > MAX_SECONDS:
                stamp(f'MAX_SECONDS={MAX_SECONDS:.0f} exceeded; terminating process group')
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)
                break
            time.sleep(2)
    finally:
        stamp(f'watchdog finished after {(time.monotonic() - started) / 60:.1f} min')

    if stopped_for_target:
        # cli.py had already marked the next stage 'running' before we terminated it.
        # 'running' is not 'done', so cli.py would re-run it anyway; normalise it to
        # 'pending' so the saved state is not misleading.
        for path in sorted((ROOT / 'work').glob('*/status.json')):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            changed = [k for k, v in data.items() if v == 'running']
            for key in changed:
                data[key] = 'pending'
            if changed:
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
                tmp.replace(path)
                stamp(f'normalised {changed} back to pending in {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
