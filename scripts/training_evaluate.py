"""
Generate training diagnostic plots from Stable-Baselines3 TensorBoard logs.

Default behavior:
- analyze the current experiment from settings.py
- if that timestamped run does not contain TensorBoard logs yet,
  fall back to the latest compatible prior run in artifacts/

Optional behavior:
- analyze a specific existing experiment by passing --artifact-dir

Expected TensorBoard location:
- <artifact_dir>/tb_logs/

Saved outputs:
- one PNG per metric in training_diagnostics/plots/
- one combined overview figure in training_diagnostics/plots/
- one CSV per metric in training_diagnostics/csv/
- one merged long-format CSV in training_diagnostics/csv/
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Ensure Python can find project folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise ImportError(
        "tensorboard is required for training_evaluate.py.\n"
        "Install it with: pip install tensorboard"
    ) from exc


SCALAR_TAGS = {
    "rollout/ep_rew_mean": "episode_reward_mean",
    "rollout/ep_len_mean": "episode_length_mean",
    "train/entropy_loss": "entropy_loss",
    "train/value_loss": "value_loss",
    "train/explained_variance": "explained_variance",
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/loss": "total_loss",
    "train/policy_gradient_loss": "policy_gradient_loss",
}

PREFERRED_OVERVIEW_ORDER = [
    "episode_reward_mean",
    "episode_length_mean",
    "entropy_loss",
    "value_loss",
    "explained_variance",
    "approx_kl",
    "clip_fraction",
    "total_loss",
    "policy_gradient_loss",
]


def _find_event_files(tb_log_dir: Path) -> list[Path]:
    """
    Recursively find TensorBoard event files inside a tb_logs directory.
    """
    if not tb_log_dir.exists():
        return []

    return sorted(
        p for p in tb_log_dir.rglob("*")
        if p.is_file() and "events.out.tfevents" in p.name
    )


def _resolve_training_artifact_dir(explicit_artifact_dir: str | None) -> Path:
    """
    Resolve the artifact directory to evaluate.

    Behavior:
    1) If --artifact-dir is provided, use it directly.
    2) Otherwise try the current timestamped settings.ARTIFACT_DIR.
    3) If current run has no TensorBoard logs, fall back to the latest compatible run
       in settings.ARTIFACTS_BASE_DIR, mirroring backtest.py logic.
    """
    if explicit_artifact_dir:
        artifact_dir = Path(explicit_artifact_dir).resolve()
        if not artifact_dir.exists():
            raise FileNotFoundError(f"❌ Provided artifact dir does not exist: {artifact_dir}")
        return artifact_dir

    current_artifact_dir = Path(settings.ARTIFACT_DIR).resolve()
    current_tb_dir = current_artifact_dir / "tb_logs"
    current_events = _find_event_files(current_tb_dir)

    if current_events:
        return current_artifact_dir

    prefix = (
        f"ppo_{settings.SYMBOL}_{settings.TIMEFRAME}_{settings.ACTION_SPACE_TYPE}"
        f"_{'news' if settings.USE_NEWS_FEATURES else 'nonews'}"
        f"_{'macro' if settings.USE_MACRO_FEATURES else 'nomacro'}"
        f"_{'time' if settings.USE_TIME_FEATURES else 'notime'}_"
    )

    candidates = []
    base_dir = Path(settings.ARTIFACTS_BASE_DIR).resolve()

    if base_dir.exists():
        for run_dir in base_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not run_dir.name.startswith(prefix):
                continue

            tb_dir = run_dir / "tb_logs"
            event_files = _find_event_files(tb_dir)
            if event_files:
                newest_event_mtime = max(p.stat().st_mtime for p in event_files)
                candidates.append((newest_event_mtime, run_dir))

    if not candidates:
        raise FileNotFoundError(
            "❌ No TensorBoard logs found for the current run and no compatible prior run found in "
            f"{settings.ARTIFACTS_BASE_DIR}.\n"
            "Make sure train.py writes TensorBoard logs into <artifact_dir>/tb_logs/."
        )

    latest_run_dir = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    print(f"ℹ️ Using latest compatible artifact run: {latest_run_dir}")
    return latest_run_dir.resolve()


def resolve_paths(artifact_dir_arg: str | None) -> tuple[Path, Path, Path, Path]:
    """
    Resolve artifact, TensorBoard log, diagnostics plot, and diagnostics csv directories.
    """
    artifact_dir = _resolve_training_artifact_dir(artifact_dir_arg)
    tb_log_dir = artifact_dir / "tb_logs"
    diagnostics_dir = artifact_dir / "training_diagnostics"
    plots_dir = diagnostics_dir / "plots"
    csv_dir = diagnostics_dir / "csv"

    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    return artifact_dir, tb_log_dir, plots_dir, csv_dir


def load_scalars_from_event_file(event_file: Path) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """
    Load supported scalar time series from one TensorBoard event file.
    """
    acc = EventAccumulator(str(event_file))
    acc.Reload()

    available_tags = set(acc.Tags().get("scalars", []))
    extracted: dict[str, pd.DataFrame] = {}

    for tb_tag, short_name in SCALAR_TAGS.items():
        if tb_tag not in available_tags:
            continue

        events = acc.Scalars(tb_tag)
        if not events:
            continue

        extracted[short_name] = pd.DataFrame({
            "wall_time": [e.wall_time for e in events],
            "step": [e.step for e in events],
            "value": [e.value for e in events],
        })

    return extracted, available_tags


def merge_scalar_runs(event_files: list[Path]) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """
    Merge supported scalars across all event files.
    """
    merged: dict[str, list[pd.DataFrame]] = {}
    all_available_tags: set[str] = set()

    for event_file in event_files:
        scalars, available_tags = load_scalars_from_event_file(event_file)
        all_available_tags.update(available_tags)

        for metric_name, df_metric in scalars.items():
            merged.setdefault(metric_name, []).append(df_metric)

    final: dict[str, pd.DataFrame] = {}
    for metric_name, dfs in merged.items():
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        final[metric_name] = df.reset_index(drop=True)

    return final, all_available_tags


def save_scalar_csvs(metrics: dict[str, pd.DataFrame], csv_dir: Path) -> None:
    """
    Save one CSV per metric plus one merged long-format CSV.
    """
    long_rows = []

    for metric_name, df_metric in metrics.items():
        df_metric.to_csv(csv_dir / f"{metric_name}.csv", index=False)

        df_long = df_metric.copy()
        df_long["metric"] = metric_name
        long_rows.append(df_long)

    if long_rows:
        df_all = pd.concat(long_rows, ignore_index=True)
        df_all = df_all[["metric", "wall_time", "step", "value"]]
        df_all.to_csv(csv_dir / "training_metrics_long.csv", index=False)


def plot_single_metric(
    df_metric: pd.DataFrame,
    metric_name: str,
    save_path: Path,
    experiment_name: str,
) -> None:
    """
    Save one diagnostic plot for a single metric.
    """
    if df_metric.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_metric["step"], df_metric["value"], linewidth=1.5)
    ax.set_title(f"{metric_name} - {experiment_name}")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_combined_overview(
    metrics: dict[str, pd.DataFrame],
    plots_dir: Path,
    experiment_name: str,
) -> None:
    """
    Save one combined overview plot containing all key available diagnostics.
    """
    available = [m for m in PREFERRED_OVERVIEW_ORDER if m in metrics and not metrics[m].empty]
    if not available:
        return

    n = len(available)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, metric_name in zip(axes, available):
        df_metric = metrics[metric_name]
        ax.plot(df_metric["step"], df_metric["value"], linewidth=1.4)
        ax.set_title(metric_name)
        ax.set_xlabel("Training Timesteps")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")

    fig.suptitle(f"Training Diagnostics - {experiment_name}", y=1.01)
    plt.tight_layout()
    plt.savefig(
        plots_dir / "training_diagnostics_overview.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


def print_metric_summary(metrics: dict[str, pd.DataFrame], all_available_tags: set[str]) -> None:
    """
    Print a short summary of extracted and missing metrics.
    """
    print("\n" + "=" * 80)
    print("TRAINING DIAGNOSTIC SUMMARY")
    print("=" * 80)

    if not metrics:
        print("No supported scalar metrics were extracted.")
    else:
        print("Extracted metrics:")
        for metric_name, df_metric in metrics.items():
            step_min = int(df_metric["step"].min())
            step_max = int(df_metric["step"].max())
            last_value = df_metric["value"].iloc[-1]
            print(
                f"- {metric_name}: {len(df_metric)} points "
                f"(steps {step_min} -> {step_max}, last={last_value:.4f})"
            )

    expected_tags = set(SCALAR_TAGS.keys())
    missing_tags = sorted(expected_tags - all_available_tags)

    if missing_tags:
        print("\nExpected TensorBoard tags not found:")
        for tag in missing_tags:
            print(f"- {tag}")

    if all_available_tags:
        print("\nAll available TensorBoard scalar tags:")
        for tag in sorted(all_available_tags):
            print(f"- {tag}")


def write_summary_json(
    artifact_dir: Path,
    tb_log_dir: Path,
    plots_dir: Path,
    csv_dir: Path,
    metrics: dict[str, pd.DataFrame],
    all_available_tags: set[str],
    event_files: list[Path],
) -> None:
    """
    Write a small diagnostics receipt JSON.
    """
    summary = {
        "artifact_dir": str(artifact_dir),
        "tb_log_dir": str(tb_log_dir),
        "event_files_found": [str(p) for p in event_files],
        "metrics_found": sorted(list(metrics.keys())),
        "available_tensorboard_tags": sorted(list(all_available_tags)),
        "plots_dir": str(plots_dir),
        "csv_dir": str(csv_dir),
    }

    summary_path = artifact_dir / "training_diagnostics" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    """
    Load TensorBoard logs and generate training diagnostic plots.
    """
    parser = argparse.ArgumentParser(
        description="Generate training diagnostic plots from TensorBoard logs."
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Optional path to an existing artifact folder. If omitted, the current experiment from settings.py is used, with fallback to the latest compatible prior run.",
    )
    args = parser.parse_args()

    artifact_dir, tb_log_dir, plots_dir, csv_dir = resolve_paths(args.artifact_dir)
    experiment_name = artifact_dir.name

    print("\n" + "=" * 80)
    print(f"TRAINING DIAGNOSTICS: {experiment_name}")
    print("=" * 80)
    print(f"Artifact dir:        {artifact_dir}")
    print(f"TensorBoard log dir: {tb_log_dir}")

    event_files = _find_event_files(tb_log_dir)
    if not event_files:
        print("\nNo TensorBoard event files found.")
        print("Expected location: <artifact_dir>/tb_logs/")
        print("Make sure train.py writes PPO TensorBoard logs into that folder.")
        return

    print(f"\nFound {len(event_files)} TensorBoard event file(s).")

    metrics, all_available_tags = merge_scalar_runs(event_files)
    if not metrics:
        print("\nNo supported scalar tags found in the event files.")
        print_metric_summary(metrics, all_available_tags)
        write_summary_json(
            artifact_dir=artifact_dir,
            tb_log_dir=tb_log_dir,
            plots_dir=plots_dir,
            csv_dir=csv_dir,
            metrics=metrics,
            all_available_tags=all_available_tags,
            event_files=event_files,
        )
        return

    save_scalar_csvs(metrics, csv_dir)

    for metric_name, df_metric in metrics.items():
        plot_single_metric(
            df_metric=df_metric,
            metric_name=metric_name,
            save_path=plots_dir / f"{metric_name}.png",
            experiment_name=experiment_name,
        )

    plot_combined_overview(
        metrics=metrics,
        plots_dir=plots_dir,
        experiment_name=experiment_name,
    )

    print_metric_summary(metrics, all_available_tags)

    write_summary_json(
        artifact_dir=artifact_dir,
        tb_log_dir=tb_log_dir,
        plots_dir=plots_dir,
        csv_dir=csv_dir,
        metrics=metrics,
        all_available_tags=all_available_tags,
        event_files=event_files,
    )

    print(f"\nPlot diagnostics saved to: {plots_dir}")
    print(f"CSV diagnostics saved to:  {csv_dir}")


if __name__ == "__main__":
    main()
