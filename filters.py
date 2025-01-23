import numpy as np
from scipy.fft import fft, ifft
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
    
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        fft_magnitude = np.abs(fft(window))  # Compute FFT magnitude
        fft_results.append(fft_magnitude[window_size - step_size:window_size])

    return np.array(fft_results)

def expanding_mean_centering(sig):
    expanding_mean = np.cumsum(sig) / np.arange(1, len(sig) + 1)
    return (sig - expanding_mean).reshape(-1, 1)

def expanding_standardization(sig):
    expanding_mean = np.cumsum(sig) / np.arange(1, len(sig) + 1)
    expanding_var = (np.cumsum((sig - expanding_mean) ** 2) / np.arange(1, len(sig) + 1))
    expanding_std = np.sqrt(expanding_var)
    return ((sig - expanding_mean) / np.maximum(expanding_std, 1e-8)).reshape(-1, 1)  # Avoid division by zero

def expanding_minmax_normalization(signal):
    normalized = []
    for t in range(1, len(signal) + 1):
        min_val = np.min(signal[:t])
        max_val = np.max(signal[:t])
        normalized.append((signal[t-1] - min_val) / (max_val - min_val) if max_val > min_val else 0)
    return (np.array(normalized)).reshape(-1, 1)

def expanding_rolling_mean(signal):
    rolling_mean = []
    for t in range(1, len(signal) + 1):
        rolling_mean.append(np.mean(signal[:t]))
    return (np.array(rolling_mean)).reshape(-1, 1)

def log_transform(signal):
    return (np.log1p(np.abs(signal))).reshape(-1, 1)  # log(1 + |x|) to handle negatives and zeros

def sqrt_transform(signal):
    return (np.sqrt(np.abs(signal))).reshape(-1, 1)

def first_order_diff(signal):
    return (np.diff(signal, prepend=signal[0])).reshape(-1, 1)

def second_order_diff(signal):
    return (np.diff(np.diff(signal, prepend=signal[0]), prepend=signal[0])).reshape(-1, 1)

def boxcox_transform(signal):
    return (boxcox(signal - np.min(signal) + 1)[0] if (signal > 0).all() else signal).reshape(-1, 1)

def first_order_div_diff_log_interval(signal, lag=1):
    log_signal = np.log1p(1 + np.abs(signal))
    result = np.zeros_like(signal)
    result[lag:] = log_signal[lag:] - log_signal[:-lag]
    return result.reshape(-1, 1)

def second_order_div_diff_log_interval(signal, lag=1):
    first_order = first_order_div_diff_log_interval(signal, lag)
    result = np.zeros_like(first_order)
    result[lag:] = first_order[lag:] - first_order[:-lag]
    return result.reshape(-1, 1)

def kalman_filter(signal):
    n = len(signal)
    x = np.zeros(n)  # Filtered state
    p = np.zeros(n)  # Error covariance
    q = 1e-5  # Process noise variance
    r = 1e-2  # Measurement noise variance
    x[0] = signal[0]
    p[0] = 1.0
    for t in range(1, n):
        # Prediction
        x[t] = x[t - 1]
        p[t] = p[t - 1] + q
        # Update
        k = p[t] / (p[t] + r)  # Kalman gain
        x[t] = x[t] + k * (signal[t] - x[t])
        p[t] = (1 - k) * p[t]
    return x.reshape(-1, 1)

def particle_filter(signal, num_particles=100):
    n = len(signal)
    particles = np.random.normal(signal[0], 1.0, num_particles)
    weights = np.ones(num_particles) / num_particles
    estimates = []
    for t in range(n):
        # Predict
        particles += np.random.normal(0, 0.1, num_particles)
        # Update
        likelihoods = np.exp(-0.5 * ((signal[t] - particles) / 1.0) ** 2)
        weights *= likelihoods
        weights /= np.sum(weights)  # Normalize
        # Resample
        indices = np.random.choice(num_particles, num_particles, p=weights)
        particles = particles[indices]
        weights = np.ones(num_particles) / num_particles
        # Estimate
        estimates.append(np.mean(particles))
    return (np.array(estimates)).reshape(-1, 1)


def low_pass_filter(signal, cutoff_frequency, sampling_rate):

    # Perform FFT to get the frequency components
    fft_signal = fft(signal)

    # Compute the frequency axis
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)

    # Apply the low-pass filter (attenuate frequencies above cutoff)
    fft_signal[np.abs(freqs) > cutoff_frequency] = 0

    # Inverse FFT to get the filtered signal
    filtered_signal = np.fft.ifft(fft_signal)

    return np.real(filtered_signal).reshape(-1, 1)




def low_pass_filter_with_dynamic_cutoff(signal, sampling_rate, threshold=0.1):
    """
    Low-pass filter for a 1D signal with a dynamically calculated cutoff frequency.

    Parameters:
        signal (array-like): The input signal to filter.
        sampling_rate (float): The sampling rate of the signal in Hz.
        threshold (float): Fraction of the maximum amplitude in the FFT spectrum
                           used to determine the cutoff frequency.

    Returns:
        filtered_signal (array-like): The filtered signal.
        cutoff_frequency (float): The calculated cutoff frequency.
    """
    # Perform FFT to get the frequency components
    fft_signal = fft(signal)
    n = len(signal)

    # Compute the frequency axis
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_signal)

    # Find the maximum amplitude in the spectrum
    max_amplitude = np.max(magnitude_spectrum)

    # Define the cutoff frequency based on the threshold
    cutoff_frequency = np.max(freqs[magnitude_spectrum >= threshold * max_amplitude])

    # Apply the low-pass filter (attenuate frequencies above the cutoff)
    fft_signal[np.abs(freqs) > cutoff_frequency] = 0

    # Inverse FFT to get the filtered signal
    filtered_signal = ifft(fft_signal)

    return np.real(filtered_signal), cutoff_frequency


def high_pass_filter(signal, cutoff_frequency, sampling_rate):

    # Perform FFT to get the frequency components
    fft_signal = fft(signal)
    
    # Compute the frequency axis
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Apply the high-pass filter (attenuate frequencies below cutoff)
    fft_signal[np.abs(freqs) < cutoff_frequency] = 0
    
    # Inverse FFT to get the filtered signal
    filtered_signal = np.fft.ifft(fft_signal)
    
    return np.real(filtered_signal).reshape(-1, 1)


def high_pass_filter_with_dynamic_cutoff(signal, sampling_rate, threshold=0.1):
    """
    High-pass filter for a 1D signal with a dynamically calculated cutoff frequency.

    Parameters:
        signal (array-like): The input signal to filter.
        sampling_rate (float): The sampling rate of the signal in Hz.
        threshold (float): Fraction of the maximum amplitude in the FFT spectrum
                           used to determine the cutoff frequency.

    Returns:
        filtered_signal (array-like): The filtered signal.
        cutoff_frequency (float): The calculated cutoff frequency.
    """
    # Perform FFT to get the frequency components
    fft_signal = fft(signal)
    n = len(signal)

    # Compute the frequency axis
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(fft_signal)

    # Find the maximum amplitude in the spectrum
    max_amplitude = np.max(magnitude_spectrum)

    # Define the cutoff frequency based on the threshold
    cutoff_frequency = np.min(np.abs(freqs[magnitude_spectrum >= threshold * max_amplitude]))

    # Apply the high-pass filter (attenuate frequencies below the cutoff)
    fft_signal[np.abs(freqs) < cutoff_frequency] = 0

    # Inverse FFT to get the filtered signal
    filtered_signal = ifft(fft_signal)

    return np.real(filtered_signal), cutoff_frequency

