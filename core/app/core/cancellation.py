"""Cooperative cancellation and per-stage timeouts.

A stage is a subprocess, so it cannot be stopped unless the code that owns the
process agrees to look at a flag. The pipeline therefore hands a token down into
the running stage (through the context) and the stage runner stops its process
when the token is set. Nothing here kills work from another thread: a token that
nobody polls can only stop the *next* stage, and the pipeline says so honestly by
marking the stage it interrupted.

Tokens form a tree. Every stage attempt gets a child of the pipeline token, so a
timeout on one stage never cancels the stages that follow it, while a user cancel
on the root always reaches whatever stage is running.
"""

from __future__ import annotations

#: Reason used when a stage exceeds its own timeout rather than being cancelled by a user.
TIMEOUT_REASON = "timeout"


class StageInterrupted(RuntimeError):
    """Base class for a stage that was stopped on purpose."""


class StageCancelled(StageInterrupted):
    """The caller asked for the run to stop."""

    def __init__(self, reason: str = "cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class StageTimeout(StageInterrupted):
    """The stage did not finish inside its own deadline."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"stage exceeded its {seconds:g}s timeout")
        self.seconds = seconds


class CancelToken:
    """A cancellation flag that a stage runner can poll.

    ``request`` is the only mutator and is safe to call from a timer thread,
    which is how a stage timeout is enforced: the deadline flips the token and
    the runner that polls it stops its subprocess.
    """

    def __init__(self, parent: "CancelToken | None" = None) -> None:
        self._parent = parent
        self._reason: str | None = None
        self._timeout: float | None = None

    @property
    def cancelled(self) -> bool:
        if self._reason is not None:
            return True

        return self._parent.cancelled if self._parent is not None else False

    @property
    def reason(self) -> str | None:
        if self._reason is not None:
            return self._reason

        return self._parent.reason if self._parent is not None else None

    @property
    def timeout(self) -> float | None:
        if self._reason == TIMEOUT_REASON:
            return self._timeout

        return self._parent.timeout if self._parent is not None else None

    def request(self, reason: str = "cancelled", timeout: float | None = None) -> None:
        """Ask for the stage to stop. The first reason wins."""

        if self._reason is None:
            self._reason = reason
            self._timeout = timeout

    def child(self) -> "CancelToken":
        """A token for one stage attempt: cancelled by its parent, not by its siblings."""

        return CancelToken(parent=self)

    def raise_if_cancelled(self) -> None:
        """Turn the flag into the matching exception, so the reason is never lost."""

        if not self.cancelled:
            return

        if self.reason == TIMEOUT_REASON:
            raise StageTimeout(self.timeout or 0.0)

        raise StageCancelled(self.reason or "cancelled")
