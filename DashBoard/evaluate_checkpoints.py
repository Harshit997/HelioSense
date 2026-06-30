import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from .data import create_dataloaders
    from .model import FlareTCN
    from .train import evaluate
except ImportError:
    from data import create_dataloaders
    from model import FlareTCN
    from train import evaluate


METRIC_DIRECTIONS = {
    "loss": "min",
    "future_binary_accuracy": "max",
    "nowcast_class_accuracy": "max",
    "future_peak_class_accuracy": "max",
    "nowcast_log_mae": "min",
    "future_peak_log_mae": "min",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate every saved epoch checkpoint on the held-out test set")
    parser.add_argument("--checkpoint-dir", default="flare_tcn_project/checkpoints_1h")
    parser.add_argument("--pattern", default="epoch_*.pt")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--data-dir", default=None, help="Optional override for checkpoint training data-dir")
    parser.add_argument("--cache-path", default=None, help="Optional override for checkpoint cache path")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional evaluation batch size override")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--sort-by", default="future_peak_class_accuracy", choices=sorted(METRIC_DIRECTIONS))
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Optional quick-test limit")
    parser.add_argument("--log-interval", type=int, default=0)
    return parser.parse_args()


def checkpoint_epoch(path):
    name = path.name
    try:
        return int(name.split("epoch_")[1].split("_")[0])
    except (IndexError, ValueError):
        return 10**9


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def make_eval_args(saved_args, cli_args):
    args = SimpleNamespace(**saved_args)
    if cli_args.data_dir is not None:
        args.data_dir = cli_args.data_dir
    if cli_args.cache_path is not None:
        args.cache_path = cli_args.cache_path
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.log_interval = cli_args.log_interval
    return args


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "rank",
        "epoch",
        "checkpoint",
        "train_loss_from_name",
        "loss",
        "future_binary_accuracy",
        "nowcast_class_accuracy",
        "future_peak_class_accuracy",
        "nowcast_log_mae",
        "future_peak_log_mae",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_loss_from_name(path):
    marker = "_train_"
    if marker not in path.stem:
        return ""
    return path.stem.split(marker, 1)[1]


def main():
    cli_args = parse_args()
    checkpoint_dir = Path(cli_args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(cli_args.pattern), key=checkpoint_epoch)
    if cli_args.max_checkpoints is not None:
        checkpoints = checkpoints[: cli_args.max_checkpoints]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} matching {cli_args.pattern}")

    if cli_args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cli_args.device)
    print(f"[Setup] Device: {device}")
    print(f"[Setup] Found {len(checkpoints)} checkpoints in {checkpoint_dir}")

    first = load_checkpoint(checkpoints[0], device="cpu")
    eval_args = make_eval_args(first["args"], cli_args)
    print(f"[Data] Building test loader using cache: {eval_args.cache_path}")
    _, _, test_loader, channel_cols, engineered_cols = create_dataloaders(
        data_dir=eval_args.data_dir,
        cache_path=eval_args.cache_path,
        rebuild_cache=False,
        history_minutes=eval_args.history_minutes,
        lead_min_minutes=eval_args.lead_min_minutes,
        lead_max_minutes=eval_args.lead_max_minutes,
        sample_stride_minutes=eval_args.sample_stride_minutes,
        aggregation_seconds=getattr(eval_args, "aggregation_seconds", 60),
        batch_size=eval_args.batch_size,
        num_workers=eval_args.num_workers,
    )

    rows = []
    for idx, path in enumerate(checkpoints, start=1):
        print(f"[Eval] {idx}/{len(checkpoints)} {path.name}")
        ckpt = load_checkpoint(path, device="cpu")
        ckpt_channel_cols = ckpt.get("channel_cols", channel_cols)
        ckpt_engineered_cols = ckpt.get("engineered_cols", engineered_cols)
        if list(ckpt_channel_cols) != list(channel_cols) or list(ckpt_engineered_cols) != list(engineered_cols):
            raise ValueError(f"Feature columns in {path} do not match the current test loader")

        model = FlareTCN(num_channels=len(channel_cols), engineered_dim=len(engineered_cols)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_value = ckpt.get("epoch", checkpoint_epoch(path))
        metrics = evaluate(model, test_loader, device, eval_args, split=f"test epoch {epoch_value}")
        row = {
            "rank": "",
            "epoch": epoch_value,
            "checkpoint": str(path),
            "train_loss_from_name": train_loss_from_name(path),
            **metrics,
        }
        rows.append(row)
        metric_text = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        print(f"[Result] epoch={epoch_value} {metric_text}")

    direction = METRIC_DIRECTIONS[cli_args.sort_by]
    reverse = direction == "max"
    rows = sorted(rows, key=lambda row: float(row[cli_args.sort_by]), reverse=reverse)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    output_csv = cli_args.output_csv or str(checkpoint_dir / "test_metrics_by_epoch.csv")
    write_csv(output_csv, rows)
    best = rows[0]
    best_epoch = best["epoch"]
    best_metric = float(best[cli_args.sort_by])
    print(f"[Best] rank=1 epoch={best_epoch} {cli_args.sort_by}={best_metric:.6f}")
    print(f"[Save] Wrote metrics CSV: {output_csv}")


if __name__ == "__main__":
    main()
