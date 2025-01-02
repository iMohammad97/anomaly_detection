from scipy.fft import fft


def rolling_truncated_fft(signal, window_size=100, step_size=1):
    if len(signal) < window_size:
        raise ValueError("Signal length must be greater than or equal to the window size.")

    num_windows = (len(signal) - window_size) // step_size
    fft_results = []
    window = signal[start:start + window_size]
    ft_magnitude = np.abs(fft(window)) 
    fft_results.append(fft_magnitude)
    
    for start in range(window_size-step_size, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        fft_magnitude = np.abs(fft(window))  # Compute FFT magnitude
        fft_results.append(fft_magnitude[window_size - step_size:window_size])

    return np.array(fft_results)


