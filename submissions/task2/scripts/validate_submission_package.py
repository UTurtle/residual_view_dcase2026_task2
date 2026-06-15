from __future__ import annotations

import csv
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]

SYSTEMS = ["Kim_LUDO_task2_1", "Kim_LUDO_task2_2", "Kim_LUDO_task2_3"]
EVAL_MACHINES = [
    "BlowerDustCollector",
    "Sander",
    "SewingMachine",
    "ToothBrush",
    "ToyDrone",
]


def read_two_column_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 200:
        raise ValueError(f"{path} must have 200 rows, got {len(rows)}.")
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"{path} must have two columns.")
        float(row[1])
    return rows


def validate_system(task2_root: Path, system_id: str) -> None:
    system_dir = task2_root / system_id
    if not system_dir.is_dir():
        raise FileNotFoundError(system_dir)

    meta_path = system_dir / f"{system_id}.meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    label = meta["submission"]["label"]
    if label != system_id:
        raise ValueError(f"{meta_path} label={label!r}, expected {system_id!r}.")

    for machine in EVAL_MACHINES:
        anomaly_path = system_dir / f"anomaly_score_{machine}_section_00_test.csv"
        decision_path = system_dir / f"decision_result_{machine}_section_00_test.csv"
        read_two_column_csv(anomaly_path)
        decision_rows = read_two_column_csv(decision_path)
        decisions = [float(row[1]) for row in decision_rows]
        if any(value not in (0.0, 1.0) for value in decisions):
            raise ValueError(f"{decision_path} must contain only 0/1 decisions.")
        print(
            f"[ok] {system_id} {machine}: "
            f"anomaly_rows=200 decision_rows=200 decision_ones={sum(decisions):.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task2_root",
        type=Path,
        default=ROOT / "submissions" / "task2",
        help="Directory that contains Kim_LUDO_task2_* system folders.",
    )
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEMS),
        help="Comma-separated system labels to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task2_root = args.task2_root
    systems = [system.strip() for system in args.systems.split(",") if system.strip()]
    labels = []
    for system_id in systems:
        validate_system(task2_root, system_id)
        meta = yaml.safe_load(
            (task2_root / system_id / f"{system_id}.meta.yaml").read_text(
                encoding="utf-8"
            )
        )
        labels.append(meta["submission"]["label"])
    if len(set(labels)) != len(labels):
        raise ValueError(f"Submission labels must be unique: {labels}")
    print("[ok] submission package validation passed")


if __name__ == "__main__":
    main()
