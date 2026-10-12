from dataclasses import dataclass
from enum import Enum


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # Stopped on purpose (user cancel or a stage timeout). Not completed, so
    # dependents stay blocked and the saved state resumes this stage later.
    CANCELLED = "cancelled"


@dataclass
class StageState:
    stage: str
    status: StageStatus = StageStatus.NOT_STARTED
    progress: float = 0.0
    message: str = ""
