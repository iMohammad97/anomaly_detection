import numpy as np

def basic_mc(n_features, random_seed):
    normalize_windows = False
    # normalize_windows = True

    mc = {}
    mc['window_size'] = 64*4 # 64
    mc['window_step'] = 64*4
    mc['n_features'] = n_features
    mc['hidden_multiplier'] = 32
    mc['max_filters'] = 256
    mc['kernel_multiplier'] = 1
    mc['z_size'] = 20
    mc['z_size_up'] = 5
    mc['window_hierarchy'] = 1 #4
    mc['z_iters'] = 25
    mc['z_sigma'] = 0.25
    mc['z_step_size'] = 0.1
    mc['z_with_noise'] = False
    mc['z_persistent'] = True
    mc['z_iters_inference'] = 500
    mc['batch_size'] = 4
    mc['learning_rate'] = 1e-3
    mc['noise_std'] = 0.001
    mc['n_iterations'] = 1000
    mc['normalize_windows'] = normalize_windows
    mc['random_seed'] = random_seed
    mc['device'] = None

    return mc

def de_unfold(x_windows, mask_windows, window_step):
    """
    x_windows of shape (n_windows, n_features, 1, window_size)
    mask_windows of shape (n_windows, n_features, 1, window_size)
    """
    n_windows, n_features, _, window_size = x_windows.shape

    assert (window_step == 1) or (window_step == window_size), 'Window step should be either 1 or equal to window_size'

    len_series = (n_windows)*window_step + (window_size-window_step)

    x = np.zeros((len_series, n_features))
    mask = np.zeros((len_series, n_features))

    n_windows = len(x_windows)
    for i in range(n_windows):
        x_window = x_windows[i,:,0,:]
        x_window = np.swapaxes(x_window,0,1)
        x[i*window_step:(i*window_step+window_size),:] += x_window

        mask_window = mask_windows[i,:,0,:]
        mask_window = np.swapaxes(mask_window,0,1)
        mask[i*window_step:(i*window_step+window_size),:] += mask_window

    division_safe_mask = mask.copy()
    division_safe_mask[division_safe_mask==0]=1
    x = x/division_safe_mask
    mask = 1*(mask>0)
    return x, mask
