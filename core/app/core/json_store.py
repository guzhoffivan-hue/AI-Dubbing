import json
from pathlib import Path
from typing import Any


class JsonStore:
    def save(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def exists(self, path: Path) -> bool:
        return path.exists()