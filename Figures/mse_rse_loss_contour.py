import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Time vector
t = np.linspace(0, 1, 1000)

# True signal
true_signal = 4 * np.sin(5 * np.pi * 2 * t)
true_mean = np.mean(true_signal)

# Amplitude and frequency ranges
amplitudes = np.linspace(0.1, 5, 20)
frequencies = np.linspace(1, 20, 20)
F, A = np.meshgrid(frequencies, amplitudes)

# Initialize values
mse_values = np.zeros_like(A)
rse_values = np.zeros_like(A)
mse_rse_values = np.zeros_like(A)

# Compute metrics
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        test_signal = A[i, j] * np.sin(2 * np.pi * F[i, j] * t)
        mse = mean_squared_error(true_signal, test_signal)**2
        numerator = (true_signal - test_signal) ** 2
        denominator = (true_signal - true_mean) ** 2 + 1e-8
        rse = np.mean(numerator / denominator)

        mse_values[i, j] = mse
        rse_values[i, j] = rse
        mse_rse_values[i, j] = (mse + rse) / 2

# Helper function to plot and save
def plot_contour(Z, title, filename, cmap='viridis'):
    plt.figure(figsize=(10, 8))
    plt.contourf(F, A, Z, cmap=cmap, levels=100)
    plt.colorbar(label=title)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.xticks([5, 10, 15, 20])
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

# Plot and save all 3 contours
plot_contour(mse_values, 'MSE', 'mse_contour.pdf', cmap='viridis')
plot_contour(rse_values, 'RSE', 'rse_contour.pdf', cmap='plasma')
plot_contour(mse_rse_values, 'MSE + RSE', 'mse_rse_contour.pdf', cmap='cividis')
