import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("data/embeddings/umap_runs.json")


def next_run_id() -> int:
    runs = _load()
    return (runs[-1]["run"] + 1) if runs else 1


def log_run(run_id: int, folder_a: str, folder_b: str, save_path: str) -> None:
    runs = _load()
    runs.append({
        "run": run_id,
        "folder_a": folder_a,
        "folder_b": folder_b,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": Path(save_path).name,
    })
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(runs, indent=2))


def _load() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return json.loads(LOG_PATH.read_text())
