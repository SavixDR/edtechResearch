import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {Path(sys.argv[0]).name} <kc_model.json>")

    json_path = Path(sys.argv[1])
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    tasks = data.get("assignment", {}).get("tasks", [])
    if not tasks:
        raise SystemExit(f"No tasks found in {json_path}")

    for task in tasks:
        task_id = task.get("task_id", "")
        description = task.get("description", "")
        print(f"{task_id}\t{description}")


if __name__ == "__main__":
    main()
