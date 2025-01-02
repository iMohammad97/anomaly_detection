from scipy.fft import fft
import numpy as np
from scipy.fft import fft
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox

def rolling_truncated_fft(signal, window_size=100, step_size=1): # Ensure `window_size` is divisible by `step_size`.
    if len(signal) < window_size:
        raise ValueError("Signal length must be greater than or equal to the window size.")

    num_windows = (len(signal) - window_size) // step_size
    fft_results = []
    window = signal[0:window_size]
    fft_magnitude = np.abs(fft(window)) 
    for start in range(0,window_size-step_size,step_size):
        fft_results.append(fft_magnitude[start:start + step_size])
    
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        fft_magnitude = np.abs(fft(window))  # Compute FFT magnitude
        fft_results.append(fft_magnitude[window_size - step_size:window_size])

    return np.array(fft_results)

def expanding_mean_centering(sig):
    expanding_mean = np.cumsum(sig) / np.arange(1, len(sig) + 1)
    return sig - expanding_mean

def expanding_standardization(sig):
    expanding_mean = np.cumsum(sig) / np.arange(1, len(sig) + 1)
    expanding_var = (np.cumsum((sig - expanding_mean) ** 2) / np.arange(1, len(sig) + 1))
    expanding_std = np.sqrt(expanding_var)
    return (sig - expanding_mean) / np.maximum(expanding_std, 1e-8)  # Avoid division by zero

def expanding_minmax_normalization(signal):
    normalized = []
    for t in range(1, len(signal) + 1):
        min_val = np.min(signal[:t])
        max_val = np.max(signal[:t])
        normalized.append((signal[t-1] - min_val) / (max_val - min_val) if max_val > min_val else 0)
    return np.array(normalized)

def expanding_rolling_mean(signal):
    rolling_mean = []
    for t in range(1, len(signal) + 1):
        rolling_mean.append(np.mean(signal[:t]))
    return np.array(rolling_mean)

def log_transform(signal):
    return np.log1p(np.abs(signal))  # log(1 + |x|) to handle negatives and zeros

def sqrt_transform(signal):
    return np.sqrt(np.abs(signal))

def first_order_diff(signal):
    return np.diff(signal, prepend=signal[0])

def second_order_diff(signal):
    return np.diff(np.diff(signal, prepend=signal[0]), prepend=signal[0])

def boxcox_transform(signal):
    return boxcox(signal - np.min(signal) + 1)[0] if (signal > 0).all() else signal
