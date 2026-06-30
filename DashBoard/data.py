from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .utils import GOES_THRESHOLDS, MIN_FLUX, flare_class_from_flux, format_duration, safe_log10_flux
except ImportError:
    from utils import GOES_THRESHOLDS, MIN_FLUX, flare_class_from_flux, format_duration, safe_log10_flux


warnings.filterwarnings("ignore", category=PerformanceWarning)

ID_COL = "unix_time"
TARGET_COL = "xrsb_flux"
ENGINEERED_BASE_COLS = ["lc_counts_scaled", "hardness_ratio"]
TREND_COLS = [
    "xrsb_log_mean",
    "xrsb_log_max",
    "xrsb_log_diff_5m",
    "xrsb_log_diff_30m",
    "xrsb_log_roll_std_30m",
    "xrsb_log_roll_max_60m",
]


def split_feature_columns(columns):
    channel_cols = [col for col in columns if col.startswith("ch_")]
    engineered_cols = [col for col in ENGINEERED_BASE_COLS + TREND_COLS if col in columns]
    return channel_cols, engineered_cols


def _steps_for_minutes(minutes, aggregation_seconds):
    steps = int(round((minutes * 60) / aggregation_seconds))
    return max(1, steps)


def _aggregate_one_file(path, aggregation_seconds=60):
    df = pd.read_parquet(path)
    channel_cols = [col for col in df.columns if col.startswith("ch_")]
    present_engineered = [col for col in ENGINEERED_BASE_COLS if col in df.columns]
    keep_cols = [ID_COL, TARGET_COL, *present_engineered, *channel_cols]
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    df["unix_minute"] = (df[ID_COL].astype("int64") // aggregation_seconds) * aggregation_seconds

    agg = {col: "mean" for col in channel_cols + present_engineered}
    agg[TARGET_COL] = ["mean", "max"]
    minute = df.groupby("unix_minute", sort=True).agg(agg)
    minute.columns = ["_".join(col).rstrip("_") for col in minute.columns.to_flat_index()]
    rename = {f"{col}_mean": col for col in channel_cols + present_engineered}
    rename[f"{TARGET_COL}_mean"] = "xrsb_flux_mean"
    rename[f"{TARGET_COL}_max"] = "xrsb_flux_max"
    minute = minute.rename(columns=rename).reset_index().copy()
    return minute


def build_minute_cache(
    data_dir="Solex_DATASET/processed",
    cache_path="flare_tcn_project/cache/minute_dataset.parquet",
    force=False,
    aggregation_seconds=60,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        print(f"[Cache] Using existing {aggregation_seconds}s cache: {cache_path}")
        return cache_path

    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Cache] Building {aggregation_seconds}s cache from {len(files)} daily files")
    start = time.time()
    parts = []
    for idx, file in enumerate(files, start=1):
        if idx == 1 or idx == len(files) or idx % 25 == 0:
            print(f"[Cache] Aggregating {idx}/{len(files)}: {file.name}")
        parts.append(_aggregate_one_file(file, aggregation_seconds=aggregation_seconds))

    data = pd.concat(parts, ignore_index=True).sort_values("unix_minute").drop_duplicates("unix_minute", keep="last")
    data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    log_mean = safe_log10_flux(data["xrsb_flux_mean"].to_numpy())
    log_max = safe_log10_flux(data["xrsb_flux_max"].to_numpy())
    log_mean_series = pd.Series(log_mean)
    log_max_series = pd.Series(log_max)
    trend_data = pd.DataFrame(
        {
            "xrsb_log_mean": log_mean.astype(np.float32),
            "xrsb_log_max": log_max.astype(np.float32),
            "xrsb_log_diff_5m": log_mean_series.diff(_steps_for_minutes(5, aggregation_seconds)).fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_diff_30m": log_mean_series.diff(_steps_for_minutes(30, aggregation_seconds)).fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_roll_std_30m": log_mean_series.rolling(_steps_for_minutes(30, aggregation_seconds), min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_roll_max_60m": log_max_series.rolling(_steps_for_minutes(60, aggregation_seconds), min_periods=1).max().to_numpy(dtype=np.float32),
        }
    )
    data = pd.concat([data.reset_index(drop=True), trend_data], axis=1).copy()

    for col in data.columns:
        if col != "unix_minute":
            data[col] = data[col].astype(np.float32)

    data.to_parquet(cache_path, index=False)
    print(f"[Cache] Saved {len(data)} rows to {cache_path} in {format_duration(time.time() - start)}")
    return cache_path


class MinuteFlareDataset(Dataset):
    def __init__(
        self,
        frame,
        start_indices,
        channel_cols,
        engineered_cols,
        history_minutes=360,
        lead_min_minutes=720,
        lead_max_minutes=1440,
        flare_threshold=GOES_THRESHOLDS["C"],
    ):
        self.frame = frame.reset_index(drop=True)
        self.start_indices = np.asarray(start_indices, dtype=np.int64)
        self.channel_cols = channel_cols
        self.engineered_cols = engineered_cols
        self.history_minutes = history_minutes
        self.lead_min_minutes = lead_min_minutes
        self.lead_max_minutes = lead_max_minutes
        self.flare_threshold = float(flare_threshold)

        self.times = self.frame["unix_minute"].to_numpy(dtype=np.int64)
        self.channels = self.frame[channel_cols].to_numpy(dtype=np.float32)
        self.engineered = self.frame[engineered_cols].to_numpy(dtype=np.float32)
        self.current_flux = self.frame["xrsb_flux_max"].to_numpy(dtype=np.float32)
        self._precompute_targets()

    def _precompute_targets(self):
        nowcast_log = np.empty(len(self.start_indices), dtype=np.float32)
        future_peak_log = np.empty(len(self.start_indices), dtype=np.float32)
        future_label = np.empty(len(self.start_indices), dtype=np.float32)
        nowcast_class = np.empty(len(self.start_indices), dtype=np.int64)
        future_class = np.empty(len(self.start_indices), dtype=np.int64)
        sample_time = np.empty(len(self.start_indices), dtype=np.int64)

        for i, start_idx in enumerate(self.start_indices):
            t0_idx = start_idx + self.history_minutes - 1
            t0 = self.times[t0_idx]
            future_start = np.searchsorted(self.times, t0 + self.lead_min_minutes * 60, side="left")
            future_end = np.searchsorted(self.times, t0 + self.lead_max_minutes * 60, side="right")
            current_peak = float(self.current_flux[t0_idx])
            future_peak = float(np.max(self.current_flux[future_start:future_end])) if future_end > future_start else MIN_FLUX

            nowcast_log[i] = float(safe_log10_flux(current_peak))
            future_peak_log[i] = float(safe_log10_flux(future_peak))
            future_label[i] = float(future_peak >= self.flare_threshold)
            nowcast_class[i] = flare_class_from_flux(current_peak)
            future_class[i] = flare_class_from_flux(future_peak)
            sample_time[i] = t0

        self.nowcast_log = nowcast_log
        self.future_peak_log = future_peak_log
        self.future_label = future_label
        self.nowcast_class = nowcast_class
        self.future_class = future_class
        self.sample_time = sample_time

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.history_minutes
        return {
            "channels": torch.from_numpy(self.channels[start_idx:end_idx].copy()),
            "engineered": torch.from_numpy(self.engineered[start_idx:end_idx].copy()),
            "nowcast_log_flux": torch.tensor(self.nowcast_log[idx], dtype=torch.float32),
            "future_peak_log_flux": torch.tensor(self.future_peak_log[idx], dtype=torch.float32),
            "future_flare_label": torch.tensor(self.future_label[idx], dtype=torch.float32),
            "nowcast_class": torch.tensor(self.nowcast_class[idx], dtype=torch.long),
            "future_peak_class": torch.tensor(self.future_class[idx], dtype=torch.long),
            "unix_minute": torch.tensor(self.sample_time[idx], dtype=torch.long),
        }


def make_start_indices(frame, history_minutes, lead_max_minutes, sample_stride_minutes):
    times = frame["unix_minute"].to_numpy(dtype=np.int64)
    starts = []
    max_start = len(frame) - history_minutes
    last_time = times[-1]
    for start in range(0, max_start + 1, sample_stride_minutes):
        t0 = times[start + history_minutes - 1]
        if t0 + lead_max_minutes * 60 <= last_time:
            starts.append(start)
    return np.asarray(starts, dtype=np.int64)


def create_dataloaders(
    data_dir="Solex_DATASET/processed",
    cache_path="flare_tcn_project/cache/minute_dataset.parquet",
    rebuild_cache=False,
    history_minutes=360,
    lead_min_minutes=720,
    lead_max_minutes=1440,
    sample_stride_minutes=5,
    aggregation_seconds=60,
    batch_size=64,
    num_workers=0,
    train_ratio=0.7,
    val_ratio=0.15,
    flare_threshold=GOES_THRESHOLDS["C"],
):
    history_steps = _steps_for_minutes(history_minutes, aggregation_seconds)
    sample_stride_steps = _steps_for_minutes(sample_stride_minutes, aggregation_seconds)
    cache_path = build_minute_cache(data_dir, cache_path, force=rebuild_cache, aggregation_seconds=aggregation_seconds)
    frame = pd.read_parquet(cache_path).sort_values("unix_minute").reset_index(drop=True)
    channel_cols, engineered_cols = split_feature_columns(frame.columns)
    if not channel_cols:
        raise ValueError("No ch_* channel columns found")
    if not engineered_cols:
        raise ValueError("No engineered columns found")

    starts = make_start_indices(frame, history_steps, lead_max_minutes, sample_stride_steps)
    if len(starts) < 3:
        raise ValueError("Not enough samples. Reduce history/lead window or sample stride.")

    train_end = int(len(starts) * train_ratio)
    val_end = int(len(starts) * (train_ratio + val_ratio))
    train_starts = starts[:train_end]
    val_starts = starts[train_end:val_end]
    test_starts = starts[val_end:]

    print(f"[Data] Aggregation: {aggregation_seconds}s | Rows: {len(frame)}")
    print(f"[Data] History: {history_minutes} minutes = {history_steps} steps | Sample stride: {sample_stride_minutes} minutes = {sample_stride_steps} steps")
    print(f"[Data] Samples: train={len(train_starts)}, val={len(val_starts)}, test={len(test_starts)}")
    print(f"[Data] Channel cols: {len(channel_cols)} | Engineered cols: {engineered_cols}")

    common = dict(
        frame=frame,
        channel_cols=channel_cols,
        engineered_cols=engineered_cols,
        history_minutes=history_steps,
        lead_min_minutes=lead_min_minutes,
        lead_max_minutes=lead_max_minutes,
        flare_threshold=flare_threshold,
    )
    train_ds = MinuteFlareDataset(start_indices=train_starts, **common)
    val_ds = MinuteFlareDataset(start_indices=val_starts, **common)
    test_ds = MinuteFlareDataset(start_indices=test_starts, **common)

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
        channel_cols,
        engineered_cols,
    )
