import json
import os
from typing import Any


class JSONLinesLogger:
    def __init__(self, filepath: str = "logs/grok_run.jsonl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            pass

    def log(self, step: int, data: dict[str, Any]):
        record = {"step": step, **data}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")