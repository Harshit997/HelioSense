import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .utils import torch_class_from_log_flux
except ImportError:
    from utils import torch_class_from_log_flux


class SpectralCNNEncoder(nn.Module):
    def __init__(self, num_channels, embedding_dim=96, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(self, x):
        batch, steps, channels = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=0.0)
        x = torch.clamp(x, min=0.0, max=1e6)
        x = torch.log1p(x)
        x = self.norm(x)
        x = x.reshape(batch * steps, 1, channels)
        x = self.net(x)
        return x.reshape(batch, steps, -1)


class EngineeredFeatureEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, min=-1e6, max=1e6)
        return self.net(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, steps, channels)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class FlareTCN(nn.Module):
    def __init__(
        self,
        num_channels,
        engineered_dim,
        spectral_embedding_dim=96,
        engineered_embedding_dim=64,
        tcn_channels=160,
        tcn_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.spectral_encoder = SpectralCNNEncoder(num_channels, spectral_embedding_dim, dropout=dropout)
        self.engineered_encoder = EngineeredFeatureEncoder(engineered_dim, engineered_embedding_dim, dropout=dropout)
        fused_dim = spectral_embedding_dim + engineered_embedding_dim
        self.input_projection = nn.Linear(fused_dim, tcn_channels)
        self.tcn = nn.Sequential(
            *[TCNBlock(tcn_channels, kernel_size=5, dilation=2 ** i, dropout=dropout) for i in range(tcn_layers)]
        )
        self.pool = AttentionPooling(tcn_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(tcn_channels),
            nn.Linear(tcn_channels, tcn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.nowcast_head = nn.Linear(tcn_channels, 1)
        self.future_peak_head = nn.Linear(tcn_channels, 1)
        self.future_probability_head = nn.Linear(tcn_channels, 1)

    def forward(self, channels, engineered):
        spectral = self.spectral_encoder(channels)
        engineered = self.engineered_encoder(engineered)
        x = torch.cat([spectral, engineered], dim=-1)
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        pooled, attention = self.pool(x)
        hidden = self.head(pooled)
        nowcast_log_flux = self.nowcast_head(hidden).squeeze(-1)
        future_peak_log_flux = self.future_peak_head(hidden).squeeze(-1)
        future_logit = self.future_probability_head(hidden).squeeze(-1)
        return {
            "nowcast_log_flux": nowcast_log_flux,
            "future_peak_log_flux": future_peak_log_flux,
            "future_flare_logit": future_logit,
            "nowcast_class": torch_class_from_log_flux(nowcast_log_flux),
            "future_peak_class": torch_class_from_log_flux(future_peak_log_flux),
            "attention": attention,
        }
