import numpy as np
import torch

GOES_THRESHOLDS = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4
}

MIN_FLUX = 1e-9

def flare_class_from_flux(flux):
    """
    Converts raw flux value (float) to class index.
    6-class mapping:
    - Index 0: < 1e-8 (Background / below A)
    - Index 1: 1e-8 to 1e-7 (Class A)
    - Index 2: 1e-7 to 1e-6 (Class B)
    - Index 3: 1e-6 to 1e-5 (Class C)
    - Index 4: 1e-5 to 1e-4 (Class M)
    - Index 5: >= 1e-4 (Class X)
    """
    if flux < 1e-8:
        return 0
    elif flux < 1e-7:
        return 1
    elif flux < 1e-6:
        return 2
    elif flux < 1e-5:
        return 3
    elif flux < 1e-4:
        return 4
    else:
        return 5

def torch_class_from_log_flux(log_flux):
    """
    Converts log10 flux tensor to class indices tensor.
    Matching 6-class mapping:
    - Class 0: log_flux < -8.0
    - Class 1: -8.0 <= log_flux < -7.0
    - Class 2: -7.0 <= log_flux < -6.0
    - Class 3: -6.0 <= log_flux < -5.0
    - Class 4: -5.0 <= log_flux < -4.0
    - Class 5: log_flux >= -4.0
    """
    if torch.is_tensor(log_flux):
        classes = torch.zeros_like(log_flux, dtype=torch.long)
        classes[log_flux >= -8.0] = 1
        classes[log_flux >= -7.0] = 2
        classes[log_flux >= -6.0] = 3
        classes[log_flux >= -5.0] = 4
        classes[log_flux >= -4.0] = 5
        return classes
    else:
        lf = float(log_flux)
        if lf < -8.0:
            return 0
        elif lf < -7.0:
            return 1
        elif lf < -6.0:
            return 2
        elif lf < -5.0:
            return 3
        elif lf < -4.0:
            return 4
        else:
            return 5

def safe_log10_flux(flux):
    """
    Calculates log10 of flux values safely.
    Works for numpy arrays or floats.
    """
    if isinstance(flux, np.ndarray):
        return np.log10(np.clip(flux, MIN_FLUX, None))
    else:
        return float(np.log10(max(float(flux), MIN_FLUX)))

def format_duration(seconds):
    """Formats a duration in seconds to a human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"
