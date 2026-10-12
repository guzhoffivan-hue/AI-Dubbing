#!/usr/bin/env python3
"""Compare two engine runs of the same clip: the core chain against ``cli.py``.

Step 6 of the migration is not "the core produced a video" but "the core produced
the same video the engine produces": the same stages, the same artifacts, the same
result. This script answers that question per artifact, and it answers it honestly:
an absolute path inside a JSON document is not a difference (the two runs live in
different roots), while a different number of speech lines or a different token
timeline always is.

    python3 core/compare_runs.py \
        --a work/core_e2e_150s --b /Volumes/develop/e2e_cli_ref/work/42503442096-1-192 \
        --a-video output/core_e2e_150s/42503442096-1-192_ru.mp4 \
        --b-video /Volumes/develop/e2e_cli_ref/output/test_dub.mp4

Exit code is 0 when every artifact that both runs produced is equivalent, 1 when
something differs, and 2 when the two runs did not get far enough to be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

#: Artifacts in the order the engine produces them.
ARTIFACTS = (
    "audio.wav",
    "audio_meta.json",
    "asr.json",
    "alignment.json",
    "diarization.json",
    "speakers.json",
    "segments.json",
    "translated_segments.json",
    "glossary.json",
    "summary.json",
    "dialogue.json",
    "dialogue_segments.json",
    "voice_profiles.json",
    "tts_manifest.json",
    "sfx_timeline.json",
    "ambience.json",
    "mix.wav",
)


def digest(path: Path) -> str:
    hasher = hashlib.md5()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)

    return hasher.hexdigest()


def normalize(value, roots: list[str]):
    """Replace the run's own root with a placeholder so paths are not compared."""

    if isinstance(value, str):
        for root in roots:
            if root and root in value:
                value = value.replace(root, "<RUN>")

        return value

    if isinstance(value, dict):
        return {key: normalize(item, roots) for key, item in value.items()}

    if isinstance(value, list):
        return [normalize(item, roots) for item in value]

    return value


def describe_difference(a, b, trail: str = "", limit: int = 6) -> list[str]:
    """The first few places two parsed documents disagree."""

    found: list[str] = []

    def walk(left, right, path: str) -> None:
        if len(found) >= limit:
            return

        if type(left) is not type(right):
            found.append(f"{path}: {type(left).__name__} vs {type(right).__name__}")
            return

        if isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left:
                    found.append(f"{path}.{key}: missing in A")
                elif key not in right:
                    found.append(f"{path}.{key}: missing in B")
                else:
                    walk(left[key], right[key], f"{path}.{key}")
            return

        if isinstance(left, list):
            if len(left) != len(right):
                found.append(f"{path}: {len(left)} items vs {len(right)} items")

            for index, (item_a, item_b) in enumerate(zip(left, right)):
                walk(item_a, item_b, f"{path}[{index}]")
            return

        if left != right:
            found.append(f"{path}: {left!r} != {right!r}")

    walk(a, b, trail or "root")

    return found


def compare_json(path_a: Path, path_b: Path, roots: list[str]) -> tuple[bool, str]:
    try:
        data_a = normalize(json.loads(path_a.read_text(encoding="utf-8")), roots)
        data_b = normalize(json.loads(path_b.read_text(encoding="utf-8")), roots)
    except (OSError, ValueError) as error:
        return False, f"not comparable: {type(error).__name__}: {error}"

    if data_a == data_b:
        return True, "same JSON (absolute paths ignored)"

    differences = describe_difference(data_a, data_b)

    return False, "differs: " + "; ".join(differences[:3])


def probe_duration(path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
    except Exception as error:  # ffprobe missing or unreadable media
        return f"unreadable ({type(error).__name__})"


def compare_files(
    path_a: Path | None,
    path_b: Path | None,
    roots: list[str],
) -> tuple[str, bool, str]:
    if path_a is None or path_b is None:
        missing = "A" if path_a is None else "B"

        return "SKIP", False, f"absent in {missing}"

    if path_a.suffix == ".json":
        same, note = compare_json(path_a, path_b, roots)

        return ("SAME" if same else "DIFF"), same, note

    same = digest(path_a) == digest(path_b)

    return (
        "SAME" if same else "DIFF",
        same,
        f"{path_a.stat().st_size} vs {path_b.stat().st_size} bytes"
        + ("" if same else f"; md5 {digest(path_a)[:8]} vs {digest(path_b)[:8]}"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="work dir of run A (the core run)")
    parser.add_argument("--b", required=True, help="work dir of run B (the cli.py run)")
    parser.add_argument("--a-root", default=None, help="absolute root to ignore in run A")
    parser.add_argument("--b-root", default=None, help="absolute root to ignore in run B")
    parser.add_argument("--a-video", default=None, help="final video of run A")
    parser.add_argument("--b-video", default=None, help="final video of run B")
    arguments = parser.parse_args()

    work_a = Path(arguments.a).resolve()
    work_b = Path(arguments.b).resolve()
    project_a = Path(arguments.a_root).resolve() if arguments.a_root else work_a.parent.parent
    project_b = Path(arguments.b_root).resolve() if arguments.b_root else work_b.parent.parent
    # The two runs live in different roots and the work dir is named after the run,
    # so both have to be normalised away: a path is not a result. Longest first, or
    # the project prefix would eat the work dir inside it and leave the run name in.
    roots = sorted(
        {str(project_a), str(project_b), str(work_a), str(work_b)},
        key=len,
        reverse=True,
    )

    if not work_a.is_dir() or not work_b.is_dir():
        print(f"run dir missing: {work_a} / {work_b}", file=sys.stderr)

        return 2

    print(f"A (core): {work_a}")
    print(f"B (cli) : {work_b}")
    print()

    problems = 0
    compared = 0

    for name in ARTIFACTS:
        path_a = work_a / name
        path_b = work_b / name
        verdict, same, note = compare_files(
            path_a if path_a.exists() else None,
            path_b if path_b.exists() else None,
            roots,
        )

        if verdict == "SKIP":
            print(f"{name:<26} {verdict:<5} {note}")
            continue

        compared += 1

        if not same:
            problems += 1

        print(f"{name:<26} {verdict:<5} {note}")

    for label, raw_a, raw_b in (
        ("final video", arguments.a_video, arguments.b_video),
    ):
        path_a = Path(raw_a) if raw_a else None
        path_b = Path(raw_b) if raw_b else None

        if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
            print(f"{label:<26} SKIP  absent")
            continue

        compared += 1
        duration_a = probe_duration(path_a)
        duration_b = probe_duration(path_b)

        if duration_a != duration_b:
            problems += 1

        print(
            f"{label:<26} {'SAME' if duration_a == duration_b else 'DIFF':<5} "
            f"duration {duration_a} vs {duration_b}; "
            f"{path_a.stat().st_size} vs {path_b.stat().st_size} bytes; "
            f"md5 {digest(path_a)[:8]} vs {digest(path_b)[:8]}"
        )

    print()

    if compared == 0:
        print("nothing to compare: neither run produced artifacts yet")

        return 2

    if problems:
        print(f"{problems} of {compared} compared artifacts differ")
        return 1

    print(f"all {compared} compared artifacts are equivalent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
