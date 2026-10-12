"""Progress inside a long stage.

A stage can take hours (translation on a nine-hour video), so the pipeline cannot
report progress only as 0.0 then 1.0: the stage would look hung. The only live
signal such a stage gives is its own log, so a reader turns log lines into a
fraction. The reader is deliberately forgiving -- a line that does not match is
not progress, a total that never appears simply means no fraction yet -- because
a broken pattern must never fail a stage that is working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

#: Called with every log line a running stage produces.
LineCallback = Callable[[str], None]


@dataclass
class StageProgressUpdate:
    """What one log line says about a stage's progress."""

    fraction: float
    message: str = ""


class LogProgress:
    """Turn a stage's own log lines into a progress fraction.

    ``patterns`` are tried in order and the first match wins. A pattern may carry
    one or two integer groups: ``(done)`` is counted against the recorded total,
    ``(done, total)`` states both. ``total_pattern`` lets the stage announce its
    own total before the first unit finishes ("Splitting into 8 safe chunks").

    The reader also returns the matched line as the progress message, so a stage
    that is running says something useful instead of silently sitting at 0 %.
    """

    def __init__(
        self,
        patterns: tuple[str, ...],
        total: int | None = None,
        total_pattern: str | None = None,
    ) -> None:
        self.patterns = tuple(re.compile(pattern) for pattern in patterns)
        self.total_pattern = (
            re.compile(total_pattern) if total_pattern is not None else None
        )
        self.total = total

    def __call__(self, line: str) -> StageProgressUpdate | None:
        text = (line or "").strip()

        if not text:
            return None

        if self.total_pattern is not None:
            announced = self.total_pattern.search(text)

            if announced is not None:
                self.total = int(announced.group(1))

        for pattern in self.patterns:
            match = pattern.search(text)

            if match is None:
                continue

            groups = match.groups()
            done = int(groups[0])
            total = int(groups[1]) if len(groups) > 1 and groups[1] else self.total

            if not total or total <= 0:
                return None

            return StageProgressUpdate(
                fraction=min(1.0, max(0.0, done / total)),
                message=text,
            )

        return None
