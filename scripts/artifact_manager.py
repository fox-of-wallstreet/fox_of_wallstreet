"""Manage artifact folders: list runs, prune empty runs, and keep only latest N runs."""

import argparse
import os
import re
import shutil
from datetime import datetime

from config import settings


RUN_PATTERN = re.compile(
    r"^ppo_(?P<symbol>[A-Z]+)_(?P<timeframe>\w+)_(?P<action>discrete_[35])_"
    r"(?P<news>news|nonews)_(?P<macro>macro|nomacro)_(?P<time>time|notime)_(?P<ts>\d{8}_\d{4})$"
)


def _run_dirs(base_dir):
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and RUN_PATTERN.match(name):
            yield name, path


def _is_empty_run(run_path):
    keep_files = {"model.zip", "scaler.pkl", "metadata.json", "backtest_ledger.csv"}
    files = set(os.listdir(run_path))
    return len(files.intersection(keep_files)) == 0


def _has_model(run_path):
    return os.path.exists(os.path.join(run_path, "model.zip"))


def _parse_ts(run_name):
    match = RUN_PATTERN.match(run_name)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y%m%d_%H%M")


def list_runs(base_dir):
    rows = []
    for name, path in _run_dirs(base_dir):
        ts = _parse_ts(name)
        rows.append(
            {
                "name": name,
                "path": path,
                "timestamp": ts,
                "has_model": _has_model(path),
                "is_empty": _is_empty_run(path),
            }
        )
    rows.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)
    return rows


def delete_paths(paths, dry_run):
    if not paths:
        print("No runs selected for deletion.")
        return

    for p in paths:
        if dry_run:
            print(f"[DRY-RUN] Would delete: {p}")
        else:
            shutil.rmtree(p, ignore_errors=True)
            print(f"Deleted: {p}")


def command_list(args):
    runs = list_runs(settings.ARTIFACTS_BASE_DIR)
    if not runs:
        print("No artifact runs found.")
        return

    for row in runs:
        status = []
        status.append("model" if row["has_model"] else "no-model")
        status.append("empty" if row["is_empty"] else "non-empty")
        print(f"{row['name']} | {'/'.join(status)}")


def command_prune_empty(args):
    runs = list_runs(settings.ARTIFACTS_BASE_DIR)
    targets = [r["path"] for r in runs if r["is_empty"]]
    delete_paths(targets, args.dry_run)


def command_keep_latest(args):
    runs = list_runs(settings.ARTIFACTS_BASE_DIR)

    # Keep latest N model-containing runs and delete older model runs.
    model_runs = [r for r in runs if r["has_model"]]
    to_delete = [r["path"] for r in model_runs[args.keep:]]
    delete_paths(to_delete, args.dry_run)


def build_parser():
    parser = argparse.ArgumentParser(description="Artifact run maintenance utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List detected artifact runs")
    p_list.set_defaults(func=command_list)

    p_prune = sub.add_parser("prune-empty", help="Delete empty run folders")
    p_prune.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    p_prune.set_defaults(func=command_prune_empty)

    p_keep = sub.add_parser("keep-latest", help="Keep only latest N model runs")
    p_keep.add_argument("--keep", type=int, default=5, help="Number of latest model runs to keep")
    p_keep.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    p_keep.set_defaults(func=command_keep_latest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
