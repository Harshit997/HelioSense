import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .data import create_dataloaders
    from .model import FlareTCN
    from .utils import format_duration, torch_class_from_log_flux
except ImportError:
    from data import create_dataloaders
    from model import FlareTCN
    from utils import format_duration, torch_class_from_log_flux


def parse_args():
    parser = argparse.ArgumentParser(description="Train spectral-CNN + TCN-attention solar flare model")
    parser.add_argument("--data-dir", default="Solex_DATASET/processed")
    parser.add_argument("--cache-path", default="flare_tcn_project/cache/minute_dataset.parquet")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--aggregation-seconds", type=int, default=60, choices=[30, 60], help="Aggregate raw per-second data into this many seconds per row.")
    parser.add_argument("--history-minutes", type=int, default=360)
    parser.add_argument("--lead-min-minutes", type=int, default=720)
    parser.add_argument("--lead-max-minutes", type=int, default=1440)
    parser.add_argument("--horizon-hours", type=float, default=None, help="Shortcut for horizon-specific models, e.g. 1, 2, 3. Sets lead window automatically.")
    parser.add_argument("--horizon-window-minutes", type=int, default=60, help="Window width after --horizon-hours. Example: 1h with 60 means 60-120 min ahead.")
    parser.add_argument("--sample-stride-minutes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nowcast-weight", type=float, default=0.3)
    parser.add_argument("--future-peak-weight", type=float, default=1.0)
    parser.add_argument("--probability-weight", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--checkpoint-dir", default="flare_tcn_project/checkpoints")
    parser.add_argument("--save-path", default="flare_tcn_project/flare_tcn_final.pt")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def compute_loss(outputs, batch, nowcast_weight, future_peak_weight, probability_weight):
    nowcast_loss = F.smooth_l1_loss(outputs["nowcast_log_flux"], batch["nowcast_log_flux"])
    future_peak_loss = F.smooth_l1_loss(outputs["future_peak_log_flux"], batch["future_peak_log_flux"])
    probability_loss = F.binary_cross_entropy_with_logits(outputs["future_flare_logit"], batch["future_flare_label"])
    loss = nowcast_weight * nowcast_loss + future_peak_weight * future_peak_loss + probability_weight * probability_loss
    return loss, nowcast_loss, future_peak_loss, probability_loss


def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    total_loss = 0.0
    start = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["channels"], batch["engineered"])
        loss, now_loss, peak_loss, prob_loss = compute_loss(
            outputs, batch, args.nowcast_weight, args.future_peak_weight, args.probability_weight
        )
        if not torch.isfinite(loss):
            print(f"[Warn] Non-finite loss at train batch {batch_idx}; skipping")
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[Warn] Non-finite grad at train batch {batch_idx}; skipping")
            continue
        optimizer.step()
        total_loss += loss.item()
        if args.log_interval and (batch_idx == 1 or batch_idx == len(loader) or batch_idx % args.log_interval == 0):
            elapsed = time.time() - start
            avg = elapsed / batch_idx
            eta = avg * (len(loader) - batch_idx)
            print(
                f"[Train] epoch={epoch} batch={batch_idx}/{len(loader)} "
                f"loss={loss.item():.6e} avg={total_loss / batch_idx:.6e} "
                f"now={now_loss.item():.4e} peak={peak_loss.item():.4e} prob={prob_loss.item():.4e} "
                f"eta={format_duration(eta)}"
            )
    return total_loss / max(1, len(loader))


def evaluate(model, loader, device, args, split="val"):
    model.eval()
    total_loss = 0.0
    n = 0
    binary_correct = 0
    now_class_correct = 0
    future_class_correct = 0
    now_mae = 0.0
    future_mae = 0.0
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            outputs = model(batch["channels"], batch["engineered"])
            loss, _, _, _ = compute_loss(outputs, batch, args.nowcast_weight, args.future_peak_weight, args.probability_weight)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs["future_flare_logit"])
            binary_correct += ((probs >= 0.5) == (batch["future_flare_label"] >= 0.5)).sum().item()
            now_pred_class = torch_class_from_log_flux(outputs["nowcast_log_flux"])
            future_pred_class = torch_class_from_log_flux(outputs["future_peak_log_flux"])
            now_class_correct += (now_pred_class == batch["nowcast_class"]).sum().item()
            future_class_correct += (future_pred_class == batch["future_peak_class"]).sum().item()
            now_mae += torch.abs(outputs["nowcast_log_flux"] - batch["nowcast_log_flux"]).sum().item()
            future_mae += torch.abs(outputs["future_peak_log_flux"] - batch["future_peak_log_flux"]).sum().item()
            n += batch["future_flare_label"].numel()
            if args.log_interval and (batch_idx == 1 or batch_idx == len(loader) or batch_idx % args.log_interval == 0):
                elapsed = time.time() - start
                avg = elapsed / batch_idx
                eta = avg * (len(loader) - batch_idx)
                print(f"[{split}] batch={batch_idx}/{len(loader)} eta={format_duration(eta)}")
    return {
        "loss": total_loss / max(1, len(loader)),
        "future_binary_accuracy": binary_correct / max(1, n),
        "nowcast_class_accuracy": now_class_correct / max(1, n),
        "future_peak_class_accuracy": future_class_correct / max(1, n),
        "nowcast_log_mae": now_mae / max(1, n),
        "future_peak_log_mae": future_mae / max(1, n),
    }


def main():
    args = parse_args()
    if args.aggregation_seconds != 60 and args.cache_path == "flare_tcn_project/cache/minute_dataset.parquet":
        args.cache_path = f"flare_tcn_project/cache/dataset_{args.aggregation_seconds}s.parquet"
    if args.horizon_hours is not None:
        args.lead_min_minutes = int(round(args.horizon_hours * 60))
        args.lead_max_minutes = args.lead_min_minutes + args.horizon_window_minutes
        horizon_tag = f"{args.horizon_hours:g}h"
        run_tag = horizon_tag if args.aggregation_seconds == 60 else f"{horizon_tag}_{args.aggregation_seconds}s"
        if args.save_path == "flare_tcn_project/flare_tcn_final.pt":
            args.save_path = f"flare_tcn_project/flare_tcn_{run_tag}.pt"
        if args.checkpoint_dir == "flare_tcn_project/checkpoints":
            args.checkpoint_dir = f"flare_tcn_project/checkpoints_{run_tag}"
    print(f"[Setup] Aggregation: {args.aggregation_seconds}s")
    print(f"[Setup] Forecast window: {args.lead_min_minutes}-{args.lead_max_minutes} minutes after t0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    train_loader, val_loader, test_loader, channel_cols, engineered_cols = create_dataloaders(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        rebuild_cache=args.rebuild_cache,
        history_minutes=args.history_minutes,
        lead_min_minutes=args.lead_min_minutes,
        lead_max_minutes=args.lead_max_minutes,
        sample_stride_minutes=args.sample_stride_minutes,
        aggregation_seconds=args.aggregation_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = FlareTCN(num_channels=len(channel_cols), engineered_dim=len(engineered_cols)).to(device)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_train = float("inf")
    no_improve = 0
    history = []
    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"[Epoch] {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_metrics = evaluate(model, val_loader, device, args, split="val")
        improved = train_loss < best_train - args.min_delta
        if improved:
            best_train = train_loss
            no_improve = 0
        else:
            no_improve += 1
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        ckpt = checkpoint_dir / f"epoch_{epoch:03d}_train_{train_loss:.6e}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "channel_cols": channel_cols,
            "engineered_cols": engineered_cols,
            "args": vars(args),
            "history": history,
        }, ckpt)
        print(f"[Save] {ckpt}")
        print(f"[Epoch] train={train_loss:.6e} val={val_metrics} early_stop={no_improve}/{args.early_stopping_patience}")
        if no_improve >= args.early_stopping_patience:
            print(f"[EarlyStop] Training loss did not improve for {args.early_stopping_patience} epochs")
            break

    test_metrics = evaluate(model, test_loader, device, args, split="test")
    final = {
        "model_state_dict": model.state_dict(),
        "channel_cols": channel_cols,
        "engineered_cols": engineered_cols,
        "args": vars(args),
        "history": history,
        "test_metrics": test_metrics,
    }
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, args.save_path)
    print(f"[Done] Test metrics: {test_metrics}")
    print(f"[Done] Saved final checkpoint to {args.save_path}")
    print(f"[Done] Total time: {format_duration(time.time() - total_start)}")


if __name__ == "__main__":
    main()
