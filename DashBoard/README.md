# Flare TCN Project

This project trains a solar flare nowcast/forecast model using configurable 1-minute or 30-second aggregated data.

## Architecture

```text
ch_013...ch_339 -> spectral CNN per minute
lc_counts_scaled, hardness_ratio, xrsb trend features -> MLP
fused sequence -> TCN -> attention pooling
heads -> current log10 flux, future peak log10 flux, future flare probability
```

## Default Task

- Input history: last 6 hours at 1-minute resolution by default; 30-second aggregation is also supported
- Forecast target: peak `xrsb_flux` between 12 and 24 hours after t0
- Flare probability target: future peak >= C-class threshold (`1e-6`)

## Run

From the repo root:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --epochs 20 \
  --batch-size 64 \
  --num-workers 2 \
  --sample-stride-minutes 5 \
  --log-interval 50
```

To rebuild the 1-minute cache:

```bash
conda run -n L1 python flare_tcn_project/train.py --rebuild-cache
```

For faster experimentation:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --epochs 5 \
  --batch-size 64 \
  --sample-stride-minutes 15 \
  --history-minutes 360
```


## Aggregation Resolution

The raw parquet files are per-second. By default the project aggregates to 60-second rows. To try 30-second aggregation, use `--aggregation-seconds 30`.

Important: `--history-minutes`, `--lead-min-minutes`, `--lead-max-minutes`, and `--sample-stride-minutes` still mean real minutes. With 30-second aggregation, `--history-minutes 360` becomes 720 time steps instead of 360.

Example 30-second 1h forecast run:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --aggregation-seconds 30 \
  --horizon-hours 1 \
  --epochs 50 \
  --early-stopping-patience 5 \
  --min-delta 0.0001
```

This writes a separate cache and model by default:

```text
flare_tcn_project/cache/dataset_30s.parquet
flare_tcn_project/checkpoints_1h_30s/
flare_tcn_project/flare_tcn_1h_30s.pt
```

## Horizon-Specific Models

You can train separate models for 1h, 2h, 3h, etc. using `--horizon-hours`.

By default, `--horizon-hours H` creates a 60-minute target window:

```text
H=1 -> forecast peak/flare between 60 and 120 minutes after t0
H=2 -> forecast peak/flare between 120 and 180 minutes after t0
H=3 -> forecast peak/flare between 180 and 240 minutes after t0
```

Examples:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --horizon-hours 1 \
  --epochs 10 \
  --batch-size 64 \
  --sample-stride-minutes 5
```

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --horizon-hours 2 \
  --epochs 10 \
  --batch-size 64 \
  --sample-stride-minutes 5
```

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --horizon-hours 3 \
  --epochs 10 \
  --batch-size 64 \
  --sample-stride-minutes 5
```

To make a narrower exact-ish horizon, set the window width. For example, 1h to 1h30m ahead:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --horizon-hours 1 \
  --horizon-window-minutes 30
```

## Early Stopping

Training includes early stopping by default. It monitors training loss and stops when it does not improve for the configured patience.

Defaults:

```text
early stopping patience = 5
min delta = 0.0
monitor = train_loss
```

Example:

```bash
conda run -n L1 python flare_tcn_project/train.py \
  --horizon-hours 1 \
  --epochs 50 \
  --early-stopping-patience 5 \
  --min-delta 0.0001
```

Every epoch is also saved to the checkpoint folder:

```text
flare_tcn_project/checkpoints_1h/epoch_001_train_....pt
```

## Evaluate Saved Epoch Checkpoints

After training, every epoch checkpoint can be evaluated on the held-out test set using `evaluate_checkpoints.py`.

For the 1-hour model:

```bash
conda run -n L1 python flare_tcn_project/evaluate_checkpoints.py \
  --checkpoint-dir flare_tcn_project/checkpoints_1h \
  --batch-size 32 \
  --num-workers 0
```

For 2-hour and 3-hour models:

```bash
conda run -n L1 python flare_tcn_project/evaluate_checkpoints.py \
  --checkpoint-dir flare_tcn_project/checkpoints_2h \
  --batch-size 32 \
  --num-workers 0
```

```bash
conda run -n L1 python flare_tcn_project/evaluate_checkpoints.py \
  --checkpoint-dir flare_tcn_project/checkpoints_3h \
  --batch-size 32 \
  --num-workers 0
```

The script writes a ranked CSV inside the checkpoint folder:

```text
flare_tcn_project/checkpoints_1h/test_metrics_by_epoch.csv
```

By default it ranks by `future_peak_class_accuracy`. To rank by lower test loss instead:

```bash
conda run -n L1 python flare_tcn_project/evaluate_checkpoints.py \
  --checkpoint-dir flare_tcn_project/checkpoints_1h \
  --sort-by loss
```
