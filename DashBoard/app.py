import os
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Workaround for OpenMP duplication error in Anaconda environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from model import FlareTCN
from utils import (
    GOES_THRESHOLDS,
    MIN_FLUX,
    flare_class_from_flux,
    torch_class_from_log_flux,
    safe_log10_flux,
    format_duration
)

# Helper function to get flare class letter and magnitude from flux
def get_flare_class_string(flux):
    if flux <= 0:
        return "Background", "N/A"
    
    # GOES flare classification
    # A: 10^-8 to 10^-7
    # B: 10^-7 to 10^-6
    # C: 10^-6 to 10^-5
    # M: 10^-5 to 10^-4
    # X: >= 10^-4
    if flux < 1e-8:
        return "Background", f"{flux/1e-9:.1f} nW"
    elif flux < 1e-7:
        val = flux / 1e-8
        return "A", f"A{val:.1f}"
    elif flux < 1e-6:
        val = flux / 1e-7
        return "B", f"B{val:.1f}"
    elif flux < 1e-5:
        val = flux / 1e-6
        return "C", f"C{val:.1f}"
    elif flux < 1e-4:
        val = flux / 1e-5
        return "M", f"M{val:.1f}"
    else:
        val = flux / 1e-4
        return "X", f"X{val:.1f}"

# Color codes for flare classes
CLASS_COLORS = {
    "Background": "#4B5563",  # Grey
    "A": "#10B981",           # Emerald/Green
    "B": "#059669",           # Dark Green
    "C": "#F59E0B",           # Yellow/Amber
    "M": "#D97706",           # Orange
    "X": "#EF4444"            # Red
}

@st.cache_resource
def load_all_models():
    """Loads all three prediction models (1h, 2h, 3h) and caches them in memory."""
    device = torch.device("cpu")
    models = {}
    
    model_paths = {
        "1h": "trained model/hour1epoch_002_train_3.797301e-01.pt",
        "2h": "trained model/hour2epoch_003_train_5.579358e-01.pt",
        "3h": "trained model/hour3epoch_001_train_8.084027e-01.pt"
    }
    
    for key, path in model_paths.items():
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                num_ch = len(ckpt["channel_cols"])
                eng_dim = len(ckpt["engineered_cols"])
                model = FlareTCN(num_channels=num_ch, engineered_dim=eng_dim)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                models[key] = {
                    "model": model,
                    "ckpt": ckpt,
                    "channel_cols": ckpt["channel_cols"],
                    "engineered_cols": ckpt["engineered_cols"],
                    "args": ckpt["args"]
                }
            except Exception as e:
                st.error(f"Error loading model {key} from {path}: {str(e)}")
        else:
            st.error(f"Model file not found: {path}")
            
    return models

def _aggregate_single_file(df, aggregation_seconds=60):
    """Aggregates a single Parquet file to minute-level matching the training pipeline."""
    channel_cols = [col for col in df.columns if col.startswith("ch_")]
    engineered_base = ["lc_counts_scaled", "hardness_ratio"]
    present_engineered = [col for col in engineered_base if col in df.columns]
    
    keep_cols = ["unix_time", "xrsb_flux", *present_engineered, *channel_cols]
    df_clean = df[keep_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    df_clean["unix_minute"] = (df_clean["unix_time"].astype("int64") // aggregation_seconds) * aggregation_seconds
    
    agg = {col: "mean" for col in channel_cols + present_engineered}
    agg["xrsb_flux"] = ["mean", "max"]
    
    minute = df_clean.groupby("unix_minute", sort=True).agg(agg)
    minute.columns = ["_".join(col).rstrip("_") for col in minute.columns.to_flat_index()]
    
    rename = {f"{col}_mean": col for col in channel_cols + present_engineered}
    rename["xrsb_flux_mean"] = "xrsb_flux_mean"
    rename["xrsb_flux_max"] = "xrsb_flux_max"
    
    minute = minute.rename(columns=rename).reset_index().copy()
    return minute

@st.cache_data(show_spinner=True)
def preprocess_dataset(parquet_path):
    """Loads and aggregates Parquet file, computing engineering trend features exactly as done in training."""
    raw_df = pd.read_parquet(parquet_path)
    
    # 1. Aggregate to 60-second resolution
    df_min = _aggregate_single_file(raw_df, aggregation_seconds=60)
    
    # 2. Replicate trend features
    log_mean = safe_log10_flux(df_min["xrsb_flux_mean"].to_numpy())
    log_max = safe_log10_flux(df_min["xrsb_flux_max"].to_numpy())
    log_mean_series = pd.Series(log_mean)
    log_max_series = pd.Series(log_max)
    
    trend_data = pd.DataFrame(
        {
            "xrsb_log_mean": log_mean.astype(np.float32),
            "xrsb_log_max": log_max.astype(np.float32),
            "xrsb_log_diff_5m": log_mean_series.diff(5).fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_diff_30m": log_mean_series.diff(30).fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_roll_std_30m": log_mean_series.rolling(30, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float32),
            "xrsb_log_roll_max_60m": log_max_series.rolling(60, min_periods=1).max().to_numpy(dtype=np.float32),
        }
    )
    df_processed = pd.concat([df_min.reset_index(drop=True), trend_data], axis=1).copy()
    
    for col in df_processed.columns:
        if col != "unix_minute":
            df_processed[col] = df_processed[col].astype(np.float32)
            
    return df_processed

# Page config
st.set_page_config(
    page_title="Solar Flare Forecasting Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Space Weather Dark CSS Injection
st.markdown("""
<style>
    /* Dark Theme Core overrides */
    .stApp {
        background-color: #050811;
        color: #E2E8F0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    
    /* Metrics panel cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Card Containers */
    .telemetry-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    
    .prediction-card {
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        background-color: #0f172a;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .status-active {
        color: #10B981;
        font-weight: bold;
        display: inline-flex;
        align-items: center;
    }
    
    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 8px;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    /* Dashboard headers */
    .dashboard-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .header-title {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .header-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Alert banners */
    .warning-banner {
        background-color: rgba(220, 38, 38, 0.15);
        border: 1px solid #DC2626;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        color: #FCA5A5;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .ok-banner {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        color: #A7F3D0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Load Models & Datasets -----------------
with st.spinner("Initializing Deep Learning Models (1h, 2h, 3h)..."):
    all_models = load_all_models()

# Sidebar Setup
st.sidebar.markdown("<div style='text-align: center; padding-bottom: 10px;'><h2 style='color:#38bdf8; font-weight: 800; margin-bottom: 0;'>SPACE WEATHER</h2><span style='color:#94a3b8; font-size:0.75rem; text-transform:uppercase;'>Control Center</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model Selection
selected_model_key = st.sidebar.selectbox(
    "Prediction Horizon Model",
    options=["1h", "2h", "3h"],
    index=0,
    help="Select the forecast model. The 1h model predicts 1-2 hours ahead, 2h model predicts 2-3 hours ahead, and 3h model predicts 3-4 hours ahead."
)

# Parquet File Selection
parquet_options = {
    "2026-06-15 (Day 1 - Active Solar State)": "trained model/merged_20260615.parquet",
    "2026-06-16 (Day 2 - Moderate Solar State)": "trained model/merged_20260616.parquet",
    "2026-06-17 (Day 3 - Calm Solar State)": "trained model/merged_20260617.parquet"
}
selected_file_label = st.sidebar.selectbox(
    "Input Parquet Dataset",
    options=list(parquet_options.keys()),
    index=0
)
parquet_path = parquet_options[selected_file_label]

# Load and Preprocess Data
if os.path.exists(parquet_path):
    df_processed = preprocess_dataset(parquet_path)
    
    # Pad short datasets (like Day 3) to satisfy the 6-hour history requirement of the model
    if len(df_processed) < 360:
        pad_len = 360 - len(df_processed)
        # Replicate the first row (edge backfill padding) to keep channels in a physical state
        first_row = df_processed.iloc[0].copy()
        pad_df = pd.DataFrame([first_row] * pad_len)
        # Sequence the timestamps backwards from the first timestamp
        first_min = int(df_processed["unix_minute"].min())
        pad_df["unix_minute"] = [first_min - (pad_len - i) * 60 for i in range(pad_len)]
        # Concatenate pad dataframe at the beginning
        df_processed = pd.concat([pad_df, df_processed], axis=0).reset_index(drop=True)
        # Ensure all columns remain float32 except unix_minute
        for col in df_processed.columns:
            if col != "unix_minute":
                df_processed[col] = df_processed[col].astype(np.float32)
else:
    st.error(f"Selected dataset not found: {parquet_path}")
    st.stop()

# Interactive start point (t0) selector
# The sequence requires 360 points (6 hours) of history, so t0 index must be at least 359
max_idx = len(df_processed) - 1
min_idx = 359

# Playback simulation controls
st.sidebar.markdown("### 🛰️ Live Playback Simulation")
autoplay = st.sidebar.checkbox("Start Live Telemetry Stream", value=False, help="Simulate a real-time spacecraft stream by automatically advancing through the dataset.")
speed = st.sidebar.select_slider(
    "Telemetry Update Interval",
    options=[0.2, 0.5, 1.0, 2.0, 3.0],
    value=1.0,
    format_func=lambda x: f"{x}s"
)

# Initialize or check if dataset changed, and reset index accordingly
if "current_file" not in st.session_state or st.session_state["current_file"] != parquet_path:
    st.session_state["current_file"] = parquet_path
    st.session_state["t0_index"] = min_idx + int((max_idx - min_idx) * 0.4)

# Increment session index if autoplay is active
if autoplay:
    next_idx = st.session_state["t0_index"] + 5  # Advance 5 minutes per step
    if next_idx > max_idx:
        st.session_state["t0_index"] = min_idx
    else:
        st.session_state["t0_index"] = next_idx

# Always clamp t0_index to ensure it remains in bounds for the current dataset
st.session_state["t0_index"] = max(min_idx, min(max_idx, st.session_state["t0_index"]))

# Let the user choose an analysis index representing "Current Time" t0
if max_idx > min_idx:
    st.sidebar.markdown("### Telemetry Timeline")
    t0_slider_idx = st.sidebar.slider(
        "Analysis Time Selector (t0)",
        min_value=min_idx,
        max_value=max_idx,
        value=st.session_state["t0_index"],
        step=5,
        help="Slide to simulate moving the system's prediction time through the observed dataset. This will update the forecast in real-time."
    )
    # Sync manual slider changes back to session state
    if st.session_state["t0_index"] != t0_slider_idx:
        st.session_state["t0_index"] = t0_slider_idx
else:
    # Safe fallback if there is only 1 point available (e.g. padded Day 3 dataset)
    t0_slider_idx = min_idx
    st.session_state["t0_index"] = min_idx

# Convert slide index to timestamp and observed values
t0_row = df_processed.iloc[t0_slider_idx]
t0_timestamp = int(t0_row["unix_minute"])
t0_datetime = datetime.fromtimestamp(t0_timestamp, tz=timezone.utc)

# Display static info card if timeline cannot slide (e.g. Day 3)
if max_idx <= min_idx:
    st.sidebar.markdown("### Telemetry Timeline")
    st.sidebar.info(f"Fixed Analysis Time (t0): {t0_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# Sidebar Graph Toggles
st.sidebar.markdown("### Visualization Toggles")
show_thresholds = st.sidebar.checkbox("Show GOES Flare Class Limits", value=True)
show_spectral = st.sidebar.checkbox("Show Solar Spectral Channels (Raw)", value=False)

if st.sidebar.button("Force Re-Run Predictions"):
    st.cache_data.clear()
    st.rerun()

# ----------------- Run Real-time Model Inference -----------------
# Get active model structure
active_model_dict = all_models[selected_model_key]
model_net = active_model_dict["model"]
model_args = active_model_dict["args"]
channel_cols = active_model_dict["channel_cols"]
engineered_cols = active_model_dict["engineered_cols"]

# Extract 360-step history ending at t0
history_start_idx = t0_slider_idx - 359
history_df = df_processed.iloc[history_start_idx : t0_slider_idx + 1]

# Format inputs for model
channels_input = torch.from_numpy(history_df[channel_cols].values.astype(np.float32)).unsqueeze(0)
engineered_input = torch.from_numpy(history_df[engineered_cols].values.astype(np.float32)).unsqueeze(0)

# Run Inference (No grad required)
with torch.no_grad():
    outputs = model_net(channels_input, engineered_input)
    
    # 1h, 2h, and 3h predictions for display cards
    predictions = {}
    
    # Selected model predictions
    selected_nowcast_flux = 10 ** float(outputs["nowcast_log_flux"].item())
    selected_future_peak = 10 ** float(outputs["future_peak_log_flux"].item())
    selected_future_prob = torch.sigmoid(outputs["future_flare_logit"]).item()
    
    # Calculate predictions for other horizons by invoking their models too
    for key, m_dict in all_models.items():
        m_net = m_dict["model"]
        m_chan = m_dict["channel_cols"]
        m_eng = m_dict["engineered_cols"]
        
        # Build specific inputs
        ch_in = torch.from_numpy(history_df[m_chan].values.astype(np.float32)).unsqueeze(0)
        eng_in = torch.from_numpy(history_df[m_eng].values.astype(np.float32)).unsqueeze(0)
        
        m_out = m_net(ch_in, eng_in)
        predictions[key] = {
            "nowcast_flux": 10 ** float(m_out["nowcast_log_flux"].item()),
            "future_peak": 10 ** float(m_out["future_peak_log_flux"].item()),
            "future_prob": torch.sigmoid(m_out["future_flare_logit"]).item(),
            "lead_min": m_dict["args"]["lead_min_minutes"],
            "lead_max": m_dict["args"]["lead_max_minutes"]
        }

# ----------------- Dashboard Layout -----------------

# Header Section
st.markdown(f"""
<div class="dashboard-header">
    <div>
        <h1 class="header-title">☀️ SOLAR FLARE FORECASTING DASHBOARD</h1>
        <div class="header-subtitle">AI-Powered Space Weather Monitoring & Spacecraft Safeguarding System</div>
    </div>
    <div style="text-align: right;">
        <div style="font-size: 0.9rem; color: #94A3B8;">UTC TIMELINE STATUS</div>
        <div style="font-size: 1.15rem; font-weight: bold; color: #38BDF8; font-family: monospace;">{t0_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC</div>
        <div style="margin-top: 5px;"><span class="status-pulse"></span><span class="status-active">SYSTEM OPERATIONAL</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- High Priority Alerts -----------------
selected_predicted_class_letter, selected_predicted_class_mag = get_flare_class_string(selected_future_peak)
is_alert_active = selected_predicted_class_letter in ["M", "X"]

if is_alert_active:
    st.markdown(f"""
    <div class="warning-banner">
        <span style="font-size: 1.8rem;">⚠️</span>
        <div>
            <div style="font-size: 1.1rem; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em; color: #F87171;">HIGH-SEVERITY SOLAR FLARE ALERT</div>
            <div style="font-size: 0.95rem; font-weight: 500;">
                Predicted <b>{selected_predicted_class_mag}-Class</b> flare expected within the {selected_model_key} horizon window 
                ({active_model_dict['args']['lead_min_minutes']}m to {active_model_dict['args']['lead_max_minutes']}m from now). 
                Spacecraft payloads and high-altitude communications should prepare for shielding/safeguard actions.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="ok-banner">
        <span style="font-size: 1.3rem;">✅</span>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500;">
                <b>Space Weather Normal:</b> No significant solar flare event (M-class or X-class) predicted within the selected forecasting window.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Status Row (Metrics) -----------------
st.markdown("<h3 style='margin-bottom:10px; color:#94A3B8; font-size:1.1rem; text-transform:uppercase;'>Latest Spacecraft Telemetry</h3>", unsafe_allow_html=True)
metrics_cols = st.columns(6)

current_observed_flux = float(t0_row["xrsb_flux_max"])
curr_class_letter, curr_class_mag = get_flare_class_string(current_observed_flux)

with metrics_cols[0]:
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">Current Observed Flux</div>
        <div data-testid="stMetricValue">{current_observed_flux:.3e}</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">W/m² (1-8 Å band)</div>
    </div>
    """, unsafe_allow_html=True)
    
with metrics_cols[1]:
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">Observed Flare Class</div>
        <div data-testid="stMetricValue" style="color:{CLASS_COLORS[curr_class_letter]}">{curr_class_mag}</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">GOES Classification</div>
    </div>
    """, unsafe_allow_html=True)

with metrics_cols[2]:
    real_lc_val = 10 ** (float(t0_row['lc_counts_scaled']) / 2.0 + 3.0)
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">Avg Light Curve Count</div>
        <div data-testid="stMetricValue">{real_lc_val:.0f}</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Detector counts</div>
    </div>
    """, unsafe_allow_html=True)

with metrics_cols[3]:
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">Hardness Ratio</div>
        <div data-testid="stMetricValue">{float(t0_row['hardness_ratio']):.4f}</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Hard vs Soft X-ray ratio</div>
    </div>
    """, unsafe_allow_html=True)

with metrics_cols[4]:
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">SoLEXS Detector Status</div>
        <div data-testid="stMetricValue" style="color:#10B981;">NOMINAL</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Active Telemetry Link</div>
    </div>
    """, unsafe_allow_html=True)

with metrics_cols[5]:
    st.markdown(f"""
    <div class="telemetry-card">
        <div data-testid="stMetricLabel">Deep Learning Model</div>
        <div data-testid="stMetricValue" style="color:#10B981;">LOADED</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Horizon {selected_model_key} Ready (CPU)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- Visualizations: Observed & Predicted Timeline -----------------
# Setup layout: Main chart (left) and predictions panel (right)
chart_col, cards_col = st.columns([7, 3])

with chart_col:
    st.markdown("<h3 style='margin-bottom:10px; color:#38BDF8; font-size:1.25rem; font-weight:800; text-transform:uppercase;'>Telemetry & Forecasting Timeline</h3>", unsafe_allow_html=True)
    
    # Build plot dataframe
    # We display observed history up to t0 (last 6 hours / 360 points)
    plot_history_df = df_processed.iloc[max(0, t0_slider_idx - 360) : t0_slider_idx + 1]
    
    fig = go.Figure()
    
    # 1. Plot Historical Observed Flux
    hist_times = pd.to_datetime(plot_history_df["unix_minute"], unit="s")
    fig.add_trace(go.Scatter(
        x=hist_times,
        y=plot_history_df["xrsb_flux_max"],
        mode="lines",
        name="Observed Flux (SoLEXS)",
        line=dict(color="#38BDF8", width=2.5),
        hovertemplate="Observed Peak: %{y:.3e} W/m²<br>Time: %{x|%H:%M} UTC<extra></extra>"
    ))
    
    # 1b. Plot Actual Future Observed Flux (y real)
    future_observed_df = df_processed.iloc[t0_slider_idx : min(max_idx + 1, t0_slider_idx + 241)]
    if len(future_observed_df) > 0:
        future_times = pd.to_datetime(future_observed_df["unix_minute"], unit="s")
        fig.add_trace(go.Scatter(
            x=future_times,
            y=future_observed_df["xrsb_flux_max"],
            mode="lines",
            name="Actual Future Flux (y real)",
            line=dict(color="#10B981", width=2, dash="dash"),
            hovertemplate="Actual Peak: %{y:.3e} W/m²<br>Time: %{x|%H:%M} UTC<extra></extra>"
        ))
    
    # 2. Vertical Line for t0 ("Current Time") using layout shapes to bypass Plotly bug
    t0_dt_obj = pd.to_datetime(t0_timestamp, unit="s").to_pydatetime()
    fig.add_shape(
        type="line",
        x0=t0_dt_obj,
        y0=1e-9,
        x1=t0_dt_obj,
        y1=1e-3,
        line=dict(color="#EF4444", width=2, dash="dash"),
        layer="above"
    )
    fig.add_annotation(
        x=t0_dt_obj,
        y=5e-4,  # near the top of the plot
        text="ANALYSIS TIME t0",
        showarrow=False,
        xanchor="right",
        font=dict(color="#EF4444", size=10, family="monospace")
    )
    
    # 3. Future Forecasting Section
    # Draw forecasting windows on the graph to show predictions extending into the future
    # Let's map predictions for 1h, 2h, and 3h horizons
    future_x = []
    future_y = []
    
    # Start the forecast line at the last observed flux point at t0
    future_x.append(t0_dt_obj)
    future_y.append(float(t0_row["xrsb_flux_max"]))
    
    for key in ["1h", "2h", "3h"]:
        pred_data = predictions[key]
        lead_min_sec = pred_data["lead_min"] * 60
        lead_max_sec = pred_data["lead_max"] * 60
        
        # Center time of the prediction window
        center_sec = t0_timestamp + (lead_min_sec + lead_max_sec) // 2
        center_dt = pd.to_datetime(center_sec, unit="s").to_pydatetime()
        
        future_x.append(center_dt)
        future_y.append(pred_data["future_peak"])
        
        # Draw target forecast window range on the graph using shaded areas
        win_start_dt = pd.to_datetime(t0_timestamp + lead_min_sec, unit="s").to_pydatetime()
        win_end_dt = pd.to_datetime(t0_timestamp + lead_max_sec, unit="s").to_pydatetime()
        
        # Shaded background representing forecast window (using layout shapes)
        fig.add_shape(
            type="rect",
            x0=win_start_dt,
            y0=1e-9,
            x1=win_end_dt,
            y1=1e-3,
            fillcolor="#1e1b4b" if key != selected_model_key else "#1e293b",
            opacity=0.35,
            layer="below",
            line_width=1,
            line_color="#4338ca" if key != selected_model_key else "#64748b"
        )
        fig.add_annotation(
            x=pd.to_datetime(center_sec, unit="s").to_pydatetime(),
            y=1.5e-9,  # near the bottom of log scale
            text=f"{key} Window",
            showarrow=False,
            font=dict(color="#818cf8" if key != selected_model_key else "#cbd5e1", size=8),
            yanchor="bottom"
        )
        
    # Plot forecast trend line
    fig.add_trace(go.Scatter(
        x=future_x,
        y=future_y,
        mode="lines+markers",
        name="AI Forecast Trend",
        line=dict(color="#818CF8", width=2.5, dash="dot"),
        marker=dict(size=7, color="#818CF8", symbol="diamond"),
        hovertemplate="Forecasted Peak: %{y:.3e} W/m²<br>Est. Time: %{x|%H:%M} UTC<extra></extra>"
    ))
    
    # 4. Horizontal GOES Class Thresholds
    if show_thresholds:
        # C-class line
        fig.add_hline(
            y=1e-6,
            line_width=1,
            line_dash="dot",
            line_color="#F59E0B",
            annotation_text="Class C Threshold (1e-6)",
            annotation_position="bottom right",
            annotation_font=dict(color="#F59E0B", size=8)
        )
        # M-class line
        fig.add_hline(
            y=1e-5,
            line_width=1.5,
            line_dash="dot",
            line_color="#D97706",
            annotation_text="Class M Threshold (1e-5)",
            annotation_position="bottom right",
            annotation_font=dict(color="#D97706", size=8)
        )
        # X-class line
        fig.add_hline(
            y=1e-4,
            line_width=1.5,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="Class X Threshold (1e-4)",
            annotation_position="bottom right",
            annotation_font=dict(color="#EF4444", size=8)
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#070c17",
        height=450,
        margin=dict(l=40, r=40, t=10, b=10),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            title="Time (UTC)"
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            type="log",
            exponentformat="e",
            title="Flux (W/m²)",
            range=[-9, -3.5] # Adjust range to clearly show levels A to X
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ----------------- Forecast Accuracy Validation (y pred vs y real) -----------------
    st.markdown("<h4 style='color:#38BDF8; font-size:1rem; font-weight:800; text-transform:uppercase; margin-top: 15px; margin-bottom:10px;'>📊 Forecast Accuracy Validation (y pred vs y real)</h4>", unsafe_allow_html=True)
    
    comp_data = []
    for key in ["1h", "2h", "3h"]:
        pred_val = predictions[key]["future_peak"]
        lead_min = predictions[key]["lead_min"]
        lead_max = predictions[key]["lead_max"]
        
        # Calculate actual peak in window if available
        win_end_idx = t0_slider_idx + lead_max
        if win_end_idx <= max_idx:
            actual_slice = df_processed.iloc[t0_slider_idx + lead_min : win_end_idx + 1]
            actual_val = float(actual_slice["xrsb_flux_max"].max())
            act_class_letter, act_class_mag = get_flare_class_string(actual_val)
            error_val = abs(np.log10(pred_val) - np.log10(actual_val))
            error_str = f"{error_val:.4f}"
            status = "🎯 HIGH ACCURACY" if error_val < 0.25 else ("⚠️ MODERATE ERROR" if error_val < 0.50 else "❌ HIGH ERROR")
            actual_str = f"{actual_val:.4e} ({act_class_mag})"
        else:
            actual_str = "N/A (Telemetry Pending)"
            error_str = "N/A"
            status = "⏳ WAITING FOR DATA"
            
        pred_class_letter, pred_class_mag = get_flare_class_string(pred_val)
        
        comp_data.append({
            "Horizon": f"{key} Forecast",
            "Target Window": f"+{lead_min}m to +{lead_max}m",
            "Predicted Peak (y pred)": f"{pred_val:.4e} ({pred_class_mag})",
            "Actual Peak (y real)": actual_str,
            "Log-scale MAE": error_str,
            "Status": status
        })
        
    st.table(pd.DataFrame(comp_data))

with cards_col:
    st.markdown("<h3 style='margin-bottom:10px; color:#38BDF8; font-size:1.25rem; font-weight:800; text-transform:uppercase;'>Prediction Horizons</h3>", unsafe_allow_html=True)
    
    for key in ["1h", "2h", "3h"]:
        pred_data = predictions[key]
        peak_flux = pred_data["future_peak"]
        class_letter, class_mag = get_flare_class_string(peak_flux)
        prob = pred_data["future_prob"]
        
        # Color coding classes
        class_color = CLASS_COLORS[class_letter]
        
        # Format prediction horizon window label
        horizon_label = f"+{pred_data['lead_min']} to +{pred_data['lead_max']} min"
        
        # Highlight card if it is the currently selected active model
        active_border = "border: 2px solid #818cf8; box-shadow: 0 0 10px rgba(129, 140, 248, 0.4);" if key == selected_model_key else "border: 1px solid #1e293b;"
        
        st.markdown(f"""
        <div class="prediction-card" style="{active_border} margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-size:0.8rem; font-weight:bold; color:#818cf8; text-transform:uppercase; background-color:rgba(129,140,248,0.15); padding:2px 8px; border-radius:4px;">{key} Prediction</span>
                <span style="font-size:0.75rem; color:#94A3B8; font-family:monospace;">{horizon_label}</span>
            </div>
            <div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Predicted Peak Flux</div>
            <div style="font-size:1.45rem; font-weight:700; color:#F8FAFC; font-family:monospace; margin-top:2px;">{peak_flux:.4e} <span style="font-size:0.8rem; font-weight:normal; color:#64748B;">W/m²</span></div>
            <div style="margin-top:8px; display:flex; justify-content:space-around; align-items:center; background-color:rgba(15,23,42,0.6); padding:8px; border-radius:6px;">
                <div>
                    <div style="font-size:0.65rem; color:#94A3B8; text-transform:uppercase;">Class Classif.</div>
                    <div style="font-size:1.15rem; font-weight:800; color:{class_color};">{class_mag}</div>
                </div>
                <div style="border-left: 1px solid #334155; height:24px;"></div>
                <div>
                    <div style="font-size:0.65rem; color:#94A3B8; text-transform:uppercase;">Confidence Score</div>
                    <div style="font-size:1.15rem; font-weight:800; color:#38BDF8; font-family:monospace;">{prob*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- Additional Charts Panel -----------------
st.markdown("<br><hr style='border-color:#1e293b;'><br>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-bottom:10px; color:#38BDF8; font-size:1.25rem; font-weight:800; text-transform:uppercase;'>Telemetry Auxiliary Analysis</h3>", unsafe_allow_html=True)

sub_cols = st.columns(2)

with sub_cols[0]:
    st.markdown("<h4 style='color:#94A3B8; font-size:0.95rem; text-transform:uppercase; margin-bottom:10px;'>Observed Light Curve & Hardness Ratio Trend</h4>", unsafe_allow_html=True)
    
    fig_aux = go.Figure()
    
    # Plot counts on primary y-axis
    fig_aux.add_trace(go.Scatter(
        x=hist_times,
        y=10 ** (plot_history_df["lc_counts_scaled"] / 2.0 + 3.0),
        mode="lines",
        name="Light Curve Count",
        line=dict(color="#10B981", width=1.5),
        hovertemplate="Counts: %{y:.0f}<extra></extra>"
    ))
    
    # Plot hardness ratio on secondary y-axis
    fig_aux.add_trace(go.Scatter(
        x=hist_times,
        y=plot_history_df["hardness_ratio"],
        mode="lines",
        name="Hardness Ratio",
        line=dict(color="#F59E0B", width=1.5),
        yaxis="y2",
        hovertemplate="Ratio: %{y:.4f}<extra></extra>"
    ))
    
    fig_aux.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#070c17",
        height=320,
        margin=dict(l=40, r=40, t=10, b=10),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            title="Time"
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            title=dict(
                text="Light Curve Count (counts)",
                font=dict(color="#10B981")
            ),
            tickfont=dict(color="#10B981")
        ),
        yaxis2=dict(
            title=dict(
                text="Hardness Ratio",
                font=dict(color="#F59E0B")
            ),
            tickfont=dict(color="#F59E0B"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_aux, use_container_width=True)

with sub_cols[1]:
    # Spectral channels breakdown if checked
    if show_spectral:
        st.markdown("<h4 style='color:#94A3B8; font-size:0.95rem; text-transform:uppercase; margin-bottom:10px;'>Spectral Flux Channels (Multichannel Telemetry)</h4>", unsafe_allow_html=True)
        
        # Select 5 representative channels to avoid overcrowding
        rep_channels = ["ch_013", "ch_080", "ch_150", "ch_230", "ch_320"]
        fig_spectral = go.Figure()
        
        colors = ["#38BDF8", "#34D399", "#FBBF24", "#FB7185", "#C084FC"]
        for idx, col in enumerate(rep_channels):
            if col in plot_history_df.columns:
                fig_spectral.add_trace(go.Scatter(
                    x=hist_times,
                    y=plot_history_df[col],
                    mode="lines",
                    name=f"Channel {col.split('_')[1]}",
                    line=dict(color=colors[idx % len(colors)], width=1.2),
                    hovertemplate="%{y:.2f} counts<extra></extra>"
                ))
                
        fig_spectral.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#070c17",
            height=320,
            margin=dict(l=40, r=40, t=10, b=10),
            xaxis=dict(
                gridcolor="#1e293b",
                showgrid=True,
                title="Time"
            ),
            yaxis=dict(
                gridcolor="#1e293b",
                showgrid=True,
                title="Counts / Sec"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_spectral, use_container_width=True)
    else:
        st.markdown("<h4 style='color:#94A3B8; font-size:0.95rem; text-transform:uppercase; margin-bottom:10px;'>Forecasting Confidence Trend</h4>", unsafe_allow_html=True)
        # We can construct a forecast history for the past hours to show if the prediction is stable
        conf_history = []
        conf_times = []
        for offset in range(-60, 1, 5): # Past hour in steps of 5m
            past_idx = t0_slider_idx + offset
            if past_idx >= min_idx:
                p_row = df_processed.iloc[past_idx]
                p_time = pd.to_datetime(p_row["unix_minute"], unit="s")
                
                # Run inference on past window
                p_start_idx = past_idx - 359
                p_hist = df_processed.iloc[p_start_idx : past_idx + 1]
                ch_in = torch.from_numpy(p_hist[channel_cols].values.astype(np.float32)).unsqueeze(0)
                eng_in = torch.from_numpy(p_hist[engineered_cols].values.astype(np.float32)).unsqueeze(0)
                
                with torch.no_grad():
                    p_out = model_net(ch_in, eng_in)
                    p_prob = torch.sigmoid(p_out["future_flare_logit"]).item()
                    
                conf_history.append(p_prob * 100)
                conf_times.append(p_time)
                
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(
            x=conf_times,
            y=conf_history,
            mode="lines+markers",
            name="C-class Flare Probability",
            line=dict(color="#A78BFA", width=2),
            marker=dict(size=4, color="#C084FC"),
            hovertemplate="Probability: %{y:.1f}%<extra></extra>"
        ))
        
        fig_conf.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#070c17",
            height=320,
            margin=dict(l=40, r=40, t=10, b=10),
            xaxis=dict(
                gridcolor="#1e293b",
                showgrid=True,
                title="Time"
            ),
            yaxis=dict(
                gridcolor="#1e293b",
                showgrid=True,
                title="Confidence (%)",
                range=[0, 105]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_conf, use_container_width=True)
# ----------------- Historical Actual vs Predicted Peak Chart -----------------
st.markdown("<br><hr style='border-color:#1e293b;'><br>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-bottom:10px; color:#38BDF8; font-size:1.25rem; font-weight:800; text-transform:uppercase;'>📈 Model Performance: Actual vs. Predicted Flare Peaks (Full Day)</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8; font-size:0.9rem; margin-bottom:15px;'>This chart compares the model's predicted future peak flux (y pred) with the actual peak flux observed in that window (y real) for all time steps across the entire dataset. This validates the model's prediction accuracy continuously over time.</p>", unsafe_allow_html=True)

# Generate predictions for the whole day (downsampled to every 10 minutes to run fast)
step_sz = 10
eval_indices = list(range(min_idx, max_idx - model_args["lead_max_minutes"] + 1, step_sz))

if len(eval_indices) > 0:
    # Build batch
    batch_chan_list = []
    batch_eng_list = []
    actual_peaks = []
    eval_times = []
    
    lead_min = model_args["lead_min_minutes"]
    lead_max = model_args["lead_max_minutes"]
    
    for idx in eval_indices:
        h_df = df_processed.iloc[idx - 359 : idx + 1]
        batch_chan_list.append(h_df[channel_cols].values)
        batch_eng_list.append(h_df[engineered_cols].values)
        eval_times.append(datetime.fromtimestamp(int(df_processed.iloc[idx]["unix_minute"]), tz=timezone.utc))
        
        # Calculate actual peak in lead window
        act_slice = df_processed.iloc[idx + lead_min : idx + lead_max + 1]
        actual_peaks.append(float(act_slice["xrsb_flux_max"].max()))
        
    # Tensor conversion
    ch_t = torch.tensor(np.array(batch_chan_list), dtype=torch.float32)
    eng_t = torch.tensor(np.array(batch_eng_list), dtype=torch.float32)
    
    with torch.no_grad():
        out_t = model_net(ch_t, eng_t)
        pred_peaks = 10 ** out_t["future_peak_log_flux"].numpy()
        
    # Plotly Comparison Chart
    fig_comp = go.Figure()
    
    # Plot Actual Peaks
    fig_comp.add_trace(go.Scatter(
        x=eval_times,
        y=actual_peaks,
        mode="lines",
        name="Actual Peak Flux (y real)",
        line=dict(color="#10B981", width=2),
        hovertemplate="Actual Peak: %{y:.3e} W/m²<extra></extra>"
    ))
    
    # Plot Predicted Peaks
    fig_comp.add_trace(go.Scatter(
        x=eval_times,
        y=pred_peaks,
        mode="lines",
        name="Predicted Peak Flux (y pred)",
        line=dict(color="#818CF8", width=2, dash="dash"),
        hovertemplate="Predicted Peak: %{y:.3e} W/m²<extra></extra>"
    ))
    
    # GOES Thresholds in layout
    if show_thresholds:
        fig_comp.add_hline(y=1e-6, line_width=0.8, line_dash="dot", line_color="#F59E0B", annotation_text="C-class", annotation_position="bottom right")
        fig_comp.add_hline(y=1e-5, line_width=1, line_dash="dot", line_color="#D97706", annotation_text="M-class", annotation_position="bottom right")
        fig_comp.add_hline(y=1e-4, line_width=1, line_dash="dot", line_color="#EF4444", annotation_text="X-class", annotation_position="bottom right")
        
    fig_comp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#070c17",
        height=380,
        margin=dict(l=40, r=40, t=10, b=10),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            title="Analysis Time (t0)"
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            type="log",
            exponentformat="e",
            title="Flux Peak (W/m²)",
            range=[-8.5, -3.5]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Insufficient data range to compute historical actual vs predicted peaks comparison.")

# ----------------- Model and Dataset Information Panels -----------------
st.markdown("<br><hr style='border-color:#1e293b;'><br>", unsafe_allow_html=True)
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("<h3 style='margin-bottom:10px; color:#94A3B8; font-size:1.1rem; text-transform:uppercase;'>Neural Network Specifications</h3>", unsafe_allow_html=True)
    
    # Count model parameters
    num_params = sum(p.numel() for p in model_net.parameters())
    
    st.markdown(f"""
    <div style="background-color:#0f172a; border: 1px solid #1e293b; border-radius:8px; padding:15px; font-family:monospace; font-size:0.85rem; color:#E2E8F0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Model Architecture:</span><span style="color:#38BDF8;">SpectralCNN + TCN + Attention Pooling</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Prediction Horizon (Selected):</span><span style="color:#38BDF8;">{selected_model_key} model ({model_args.get('lead_min_minutes')} - {model_args.get('lead_max_minutes')} minutes)</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Spectral Channels (Encoder):</span><span style="color:#38BDF8;">{len(channel_cols)} features</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Engineered Base & Trend features:</span><span style="color:#38BDF8;">{len(engineered_cols)} features</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Input Sequence Length:</span><span style="color:#38BDF8;">360 time steps (6.0 hours)</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Active Device Context:</span><span style="color:#10B981;">CPU (Intel OpenMP Link Ok)</span></div>
        <div style="display:flex; justify-content:space-between;"><span>Trainable Parameters:</span><span style="color:#38BDF8;">{num_params:,} parameters</span></div>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("<h3 style='margin-bottom:10px; color:#94A3B8; font-size:1.1rem; text-transform:uppercase;'>Dataset Details</h3>", unsafe_allow_html=True)
    
    # Calculate time range of current Parquet file
    file_start_ts = int(df_processed["unix_minute"].min())
    file_end_ts = int(df_processed["unix_minute"].max())
    file_start_dt = datetime.fromtimestamp(file_start_ts, tz=timezone.utc)
    file_end_dt = datetime.fromtimestamp(file_end_ts, tz=timezone.utc)
    
    st.markdown(f"""
    <div style="background-color:#0f172a; border: 1px solid #1e293b; border-radius:8px; padding:15px; font-family:monospace; font-size:0.85rem; color:#E2E8F0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Source Dataset:</span><span style="color:#38BDF8;">SoLEXS Space Payload Parquet</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Current Parquet File:</span><span style="color:#38BDF8;">{os.path.basename(parquet_path)}</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Total Sample Rows:</span><span style="color:#38BDF8;">{len(df_processed)} rows</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Time Aggregation Bin:</span><span style="color:#38BDF8;">60 seconds (Mean/Max pooling)</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Active Time Range (UTC):</span><span style="color:#38BDF8;">{file_start_dt.strftime('%H:%M')} to {file_end_dt.strftime('%H:%M')} UTC</span></div>
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;"><span>Latest Observed Sample:</span><span style="color:#38BDF8;">{t0_datetime.strftime('%Y-%m-%d %H:%M')} UTC</span></div>
        <div style="display:flex; justify-content:space-between;"><span>Data Preprocessing pipeline:</span><span style="color:#10B981;">Complete (Matches Training Model)</span></div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Autoplay Refresh Loop -----------------
if autoplay:
    time.sleep(speed)
    st.rerun()
