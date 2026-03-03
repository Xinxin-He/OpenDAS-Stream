from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

try:
    from nptdms import TdmsFile
    HAS_TDMS = True
except ImportError:
    TdmsFile = None  # type: ignore[assignment]
    HAS_TDMS = False

try:
    import cupy as cp
    import cupyx.scipy.signal as cpx_sig
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count > 0:
        HAS_GPU = True
        print(f"[System] Physical GPU detected: Found {device_count} NVIDIA GPU(s).")
    else:
        raise Exception("No physical GPU")
except Exception:
    HAS_GPU = False
    import numpy as cp  # type: ignore[no-redef]
    from scipy import signal as cpx_sig  # type: ignore[assignment]
    print("[System] Physical GPU unavailable or CuPy not installed. Falling back to CPU mode.")


def get_xp(input_array):
    # CPU fallback path: cp is numpy alias and has no get_array_module.
    if not HAS_GPU:
        return np
    return cp.get_array_module(input_array)


def remove_common_mode(input_matrix, method="mean"):
    array_backend = get_xp(input_matrix)
    if not array_backend.issubdtype(input_matrix.dtype, array_backend.floating):
        input_matrix = input_matrix.astype(array_backend.float32)
    if method == "median":
        common_signal = array_backend.median(input_matrix, axis=0, keepdims=True)
    else:
        common_signal = input_matrix.mean(axis=0, keepdims=True)
    input_matrix -= common_signal
    return input_matrix


def gpu_downsample(input_matrix, channel_downsample_factor, time_downsample_factor, reduction_method="mean"):
    array_backend = get_xp(input_matrix)
    channel_count, time_count = input_matrix.shape
    if channel_downsample_factor <= 1 and time_downsample_factor <= 1:
        return input_matrix
    if not array_backend.issubdtype(input_matrix.dtype, array_backend.floating):
        input_matrix = input_matrix.astype(array_backend.float32)
    valid_channel_count = (channel_count // channel_downsample_factor) * channel_downsample_factor
    valid_time_count = (time_count // time_downsample_factor) * time_downsample_factor
    if valid_channel_count == 0 or valid_time_count == 0:
        return input_matrix
    cropped_array = input_matrix[:valid_channel_count, :valid_time_count]
    reshaped_array = cropped_array.reshape(
        valid_channel_count // channel_downsample_factor,
        channel_downsample_factor,
        valid_time_count // time_downsample_factor,
        time_downsample_factor,
    )
    if reduction_method == "max":
        downsampled_array = reshaped_array.max(axis=3).max(axis=1)
    else:
        downsampled_array = reshaped_array.mean(axis=3).mean(axis=1)
    return downsampled_array.astype(array_backend.float32)


def gpu_bandpass(
    input_matrix,
    sample_rate_hz,
    low_cut_hz,
    high_cut_hz,
    filter_order=4,
    use_zero_phase=False,
):
    array_backend = get_xp(input_matrix)
    if HAS_GPU and array_backend == np:
        input_matrix = cp.asarray(input_matrix)
        array_backend = cp
    nyquist_hz = 0.5 * sample_rate_hz
    safe_low_hz = max(1e-6, low_cut_hz)
    safe_high_hz = min(high_cut_hz, nyquist_hz * 0.9)
    if safe_low_hz >= safe_high_hz:
        return input_matrix
    sos_filter = cpx_sig.butter(
        filter_order,
        [safe_low_hz, safe_high_hz],
        btype="band",
        fs=sample_rate_hz,
        output="sos",
    )
    if HAS_GPU and array_backend == cp:
        sos_filter = cp.asarray(sos_filter)
    if use_zero_phase:
        return cpx_sig.sosfiltfilt(sos_filter, input_matrix, axis=1)
    return cpx_sig.sosfilt(sos_filter, input_matrix, axis=1)


def gpu_lowpass(input_matrix, sample_rate_hz, lowpass_cut_hz, filter_order=4, use_zero_phase=False):
    array_backend = get_xp(input_matrix)
    if HAS_GPU and array_backend == np:
        input_matrix = cp.asarray(input_matrix)
        array_backend = cp
    nyquist_hz = 0.5 * sample_rate_hz
    safe_cut_hz = max(1e-6, min(lowpass_cut_hz, nyquist_hz * 0.9))
    if safe_cut_hz >= nyquist_hz:
        return input_matrix
    sos_filter = cpx_sig.butter(filter_order, safe_cut_hz, btype="low", fs=sample_rate_hz, output="sos")
    if HAS_GPU and array_backend == cp:
        sos_filter = cp.asarray(sos_filter)
    if use_zero_phase:
        return cpx_sig.sosfiltfilt(sos_filter, input_matrix, axis=1)
    return cpx_sig.sosfilt(sos_filter, input_matrix, axis=1)


def gpu_highpass(input_matrix, sample_rate_hz, highpass_cut_hz, filter_order=2, use_zero_phase=False):
    array_backend = get_xp(input_matrix)
    if HAS_GPU and array_backend == np:
        input_matrix = cp.asarray(input_matrix)
        array_backend = cp
    nyquist_hz = 0.5 * sample_rate_hz
    safe_cut_hz = max(1e-6, min(highpass_cut_hz, nyquist_hz * 0.9))
    if safe_cut_hz >= nyquist_hz:
        return input_matrix
    sos_filter = cpx_sig.butter(filter_order, safe_cut_hz, btype="high", fs=sample_rate_hz, output="sos")
    if HAS_GPU and array_backend == cp:
        sos_filter = cp.asarray(sos_filter)
    if use_zero_phase:
        return cpx_sig.sosfiltfilt(sos_filter, input_matrix, axis=1)
    return cpx_sig.sosfilt(sos_filter, input_matrix, axis=1)


def gpu_envelope(input_matrix):
    array_backend = get_xp(input_matrix)
    return array_backend.abs(cpx_sig.hilbert(input_matrix, axis=1))


def gpu_band_rms(
    input_matrix,
    sample_rate_hz,
    low_cut_hz,
    high_cut_hz,
    smoothing_window_seconds=0.1,
    filter_order=4,
):
    filtered_array = gpu_bandpass(
        input_matrix,
        sample_rate_hz,
        low_cut_hz,
        high_cut_hz,
        filter_order=filter_order,
        use_zero_phase=True,
    )
    array_backend = get_xp(filtered_array)
    squared_array = filtered_array ** 2
    squared_array_cpu = cp.asnumpy(squared_array) if HAS_GPU and array_backend == cp else squared_array
    smooth_window_size = max(1, int(smoothing_window_seconds * sample_rate_hz))
    smoothed_array = uniform_filter1d(squared_array_cpu, size=smooth_window_size, axis=1)
    return np.sqrt(np.abs(smoothed_array)).astype(np.float32)
