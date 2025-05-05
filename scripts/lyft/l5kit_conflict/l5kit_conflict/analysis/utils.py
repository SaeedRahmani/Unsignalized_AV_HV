import numpy as np


def compute_position_based_velocities(xy: np.ndarray, dt: float = 0.1):
    """ 
    return the sequences of position-based velocities of an agent's trajectory.
    """
    assert isinstance(xy, np.ndarray), "Input's type is not np.ndarrary."
    assert xy.shape[1] == 2, "Input's shape is not valid, reshape as (N, 2)."
    assert xy.shape[0] >= 3,  "Input's size is not enough long."

    dxy = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dxy = dxy[:-1] + dxy[1:]
    v = dxy/dt/2

    assert v.shape[0] == xy.shape[0] - 2
    return v


def three_sigma_smoothing(signal, window_size=25, threshold=1.5):
    """
    Apply 3-sigma smoothing to a signal.
    
    Parameters:
        - signal: The input signal to be smoothed.
        - window_size: The size of the smoothing window.
        - threshold: The threshold in terms of standard deviations.
    
    Returns:
        - smoothed_signal: The smoothed signal.
    """
    smoothed_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        start = max(0, i - window_size//2)
        end = min(len(signal), i + window_size//2 + 1)
        window = signal[start:end]
        mean = np.mean(window)
        std = np.std(window)
        if np.abs(signal[i] - mean) > threshold * std:
            smoothed_signal[i] = mean
        else:
            smoothed_signal[i] = signal[i]
    return smoothed_signal