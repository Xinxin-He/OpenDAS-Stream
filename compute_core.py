from __future__ import annotations

from datetime import timedelta
import numpy as np
from scipy import signal
from filter_core import gpu_bandpass, gpu_lowpass, gpu_band_rms


def _to_numpy_f32(arr):
    """Convert array to numpy float32; handle CuPy arrays via .get()."""
    if hasattr(arr, "get"):
        return np.asarray(arr.get(), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def fast_percentile_sample(input_array, percentiles=(1, 99), sample_size=200_000, random_seed=0):
    random_generator = np.random.default_rng(random_seed)
    flattened_array = np.asarray(input_array).ravel()
    finite_values = flattened_array[np.isfinite(flattened_array)]
    if finite_values.size == 0:
        return 0.0, 0.0
    if finite_values.size <= sample_size:
        lower_percentile, upper_percentile = np.nanpercentile(finite_values, percentiles)
        return float(lower_percentile), float(upper_percentile)
    sampled_indices = random_generator.integers(0, finite_values.size, size=sample_size)
    sampled_values = finite_values[sampled_indices]
    lower_percentile, upper_percentile = np.nanpercentile(sampled_values, percentiles)
    return float(lower_percentile), float(upper_percentile)


def build_fbe_runtime_specs(frequency_bands, sample_rate_hz, guard_policy="clip", nyquist_clip_ratio=0.98):
    runtime_specifications = []
    nyquist_hz = 0.5 * float(sample_rate_hz)
    clipped_high_limit_hz = max(1e-6, nyquist_hz * float(nyquist_clip_ratio))
    normalized_guard_mode = str(guard_policy).lower()

    for requested_low_hz, requested_high_hz in frequency_bands:
        requested_low_hz = float(requested_low_hz)
        requested_high_hz = float(requested_high_hz)
        runtime_spec = {
            "requested_low_hz": requested_low_hz,
            "requested_high_hz": requested_high_hz,
            "runtime_low_hz": requested_low_hz,
            "runtime_high_hz": requested_high_hz,
            "is_enabled": True,
            "note_text": None,
        }
        if requested_high_hz >= nyquist_hz:
            if normalized_guard_mode == "clip":
                runtime_high_hz = min(requested_high_hz, clipped_high_limit_hz)
                runtime_low_hz = max(1e-6, min(requested_low_hz, runtime_high_hz * 0.999))
                if runtime_low_hz >= runtime_high_hz:
                    runtime_spec["is_enabled"] = False
                    runtime_spec["note_text"] = (
                        f"Band {requested_low_hz:g}-{requested_high_hz:g}Hz invalid after Nyquist guard"
                    )
                else:
                    runtime_spec["runtime_low_hz"] = runtime_low_hz
                    runtime_spec["runtime_high_hz"] = runtime_high_hz
                    runtime_spec["note_text"] = (
                        f"Band {requested_low_hz:g}-{requested_high_hz:g}Hz "
                        f"clipped to {runtime_low_hz:g}-{runtime_high_hz:g}Hz "
                        f"(Nyquist={nyquist_hz:g}Hz)"
                    )
            else:
                runtime_spec["is_enabled"] = False
                runtime_spec["note_text"] = (
                    f"Band {requested_low_hz:g}-{requested_high_hz:g}Hz exceeds Nyquist ({nyquist_hz:g}Hz)"
                )
        runtime_specifications.append(runtime_spec)

    return runtime_specifications


def blend_lfdas_boundary(previous_chunk_visual, current_chunk_visual, blend_column_count):
    blend_column_count = int(blend_column_count)
    if blend_column_count < 2:
        return
    blend_column_count = min(
        blend_column_count,
        previous_chunk_visual.shape[1],
        current_chunk_visual.shape[1],
    )
    if blend_column_count < 2:
        return
    blend_weights = np.linspace(0.0, 1.0, blend_column_count, dtype=np.float32)[None, :]
    blended_boundary = (
        previous_chunk_visual[:, -blend_column_count:] * (1.0 - blend_weights)
        + current_chunk_visual[:, :blend_column_count] * blend_weights
    )
    previous_chunk_visual[:, -blend_column_count:] = blended_boundary
    current_chunk_visual[:, :blend_column_count] = blended_boundary


def compute_raw_heatmap(raw_visual_matrix, common_mode_enabled):
    lower_percentile, upper_percentile = fast_percentile_sample(raw_visual_matrix, (2, 98))
    symmetric_limit = max(abs(lower_percentile), abs(upper_percentile))
    return {
        "visual_matrix": raw_visual_matrix,
        "colormap_name": "seismic",
        "colorbar_vmin": -symmetric_limit,
        "colorbar_vmax": symmetric_limit,
        "plot_title": "Raw DAS (Common Mode Removed)" if common_mode_enabled else "Raw DAS",
    }


def compute_freq_distance_spectrum(raw_visual_matrix, visual_sample_rate_hz, y_axis_max, y_axis_min):
    complex_spectrum = np.fft.rfft(raw_visual_matrix, axis=1)
    magnitude_spectrum = np.abs(complex_spectrum)
    frequency_axis_hz = np.fft.rfftfreq(raw_visual_matrix.shape[1], d=1.0 / visual_sample_rate_hz)
    valid_frequency_mask = frequency_axis_hz <= (visual_sample_rate_hz / 2)
    clipped_magnitude_spectrum = magnitude_spectrum[:, valid_frequency_mask]
    clipped_frequency_axis_hz = frequency_axis_hz[valid_frequency_mask]
    spectrum_extent = [clipped_frequency_axis_hz[0], clipped_frequency_axis_hz[-1], y_axis_max, y_axis_min]
    lower_percentile, upper_percentile = fast_percentile_sample(clipped_magnitude_spectrum, (2, 98))
    return {
        "spectrum_matrix": clipped_magnitude_spectrum,
        "plot_extent": spectrum_extent,
        "colorbar_vmin": lower_percentile,
        "colorbar_vmax": upper_percentile,
        "plot_title": "Frequency-Distance Spectrum",
        "x_axis_label": "Frequency (Hz)",
    }


def compute_channel_spectrogram(
    raw_visual_matrix,
    visual_sample_rate_hz,
    target_channel_row_index,
    start_datetime,
    x_axis_mode,
    time_start_seconds,
    spectrogram_nperseg=256,
    spectrogram_noverlap=128,
):
    channel_trace = raw_visual_matrix[target_channel_row_index, :]
    frequency_axis_hz, time_axis_seconds, spectrogram_power = signal.spectrogram(
        channel_trace,
        fs=visual_sample_rate_hz,
        nperseg=int(spectrogram_nperseg),
        noverlap=int(spectrogram_noverlap),
    )
    spectrogram_log_power = 10 * np.log10(spectrogram_power + 1e-9)
    if x_axis_mode == "datetime":
        time_axis = [start_datetime + timedelta(seconds=offset_seconds) for offset_seconds in time_axis_seconds]
    else:
        time_axis = time_axis_seconds + time_start_seconds
    return {
        "frequency_axis_hz": frequency_axis_hz,
        "time_axis": time_axis,
        "spectrogram_log_power": spectrogram_log_power,
    }


def compute_lfdas_map(lfdas_visual_matrix, lfdas_method, lfdas_high_hz, lfdas_low_ratio):
    lower_percentile, upper_percentile = fast_percentile_sample(lfdas_visual_matrix, (1, 99))
    lfdas_low_hz = lfdas_high_hz * lfdas_low_ratio if lfdas_method == "bandpass" else 0
    lfdas_title = (
        f"LFDAS ({lfdas_low_hz:.3g}-{lfdas_high_hz} Hz)"
        if lfdas_method == "bandpass"
        else f"LFDAS (< {lfdas_high_hz} Hz)"
    )
    return {
        "visual_matrix": lfdas_visual_matrix,
        "colormap_name": "coolwarm",
        "colorbar_vmin": lower_percentile,
        "colorbar_vmax": upper_percentile,
        "plot_title": lfdas_title,
    }


def compute_bandpass_map(band_visual_matrix, band_frequency_pair, is_envelope_band):
    lower_percentile, upper_percentile = fast_percentile_sample(band_visual_matrix, (2, 98))
    if not is_envelope_band:
        symmetric_limit = max(abs(lower_percentile), abs(upper_percentile))
        lower_percentile, upper_percentile = -symmetric_limit, symmetric_limit
    return {
        "visual_matrix": band_visual_matrix,
        "colormap_name": "inferno" if is_envelope_band else "seismic",
        "colorbar_vmin": lower_percentile,
        "colorbar_vmax": upper_percentile,
        "plot_title": f"Band {band_frequency_pair[0]}-{band_frequency_pair[1]}Hz {'(Env)' if is_envelope_band else ''}",
    }


def compute_fk_compare(
    raw_visual_matrix,
    spatial_spacing_for_fk,
    fk_pass_direction="positive",
    fk_prefilter_band_hz=None,
    sample_rate_hz=None,
    filter_order=4,
    fk_normalize_mode="none",
):
    if not (spatial_spacing_for_fk > 0 and np.isfinite(spatial_spacing_for_fk)):
        return None
    input_matrix = np.asarray(raw_visual_matrix, dtype=np.float32)
    if input_matrix.ndim != 2 or input_matrix.shape[1] == 0:
        return None

    if fk_prefilter_band_hz is not None:
        if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            return None
        prefilter_low_hz, prefilter_high_hz = float(fk_prefilter_band_hz[0]), float(fk_prefilter_band_hz[1])
        safe_low_hz, safe_high_hz = _safe_band_limits(sample_rate_hz, prefilter_low_hz, prefilter_high_hz)
        if safe_low_hz is None:
            return None
        input_matrix = np.asarray(
            gpu_bandpass(
                input_matrix,
                float(sample_rate_hz),
                safe_low_hz,
                safe_high_hz,
                filter_order=int(filter_order),
                use_zero_phase=False,
            ),
            dtype=np.float32,
        )

    temporal_spectrum = np.fft.rfft(input_matrix, axis=1)
    f_k_spectrum = np.fft.fftshift(np.fft.fft(temporal_spectrum, axis=0), axes=0)
    wavenumber_axis = np.fft.fftshift(np.fft.fftfreq(input_matrix.shape[0], d=spatial_spacing_for_fk))
    normalized_direction = str(fk_pass_direction).lower()
    if normalized_direction == "negative":
        wavenumber_mask = (wavenumber_axis <= 0)[:, None]
    elif normalized_direction == "both":
        wavenumber_mask = np.ones((input_matrix.shape[0], 1), dtype=bool)
    else:
        wavenumber_mask = (wavenumber_axis >= 0)[:, None]
    filtered_fk_spectrum = f_k_spectrum * wavenumber_mask
    inverse_temporal_spectrum = np.fft.ifft(np.fft.ifftshift(filtered_fk_spectrum, axes=0), axis=0)
    fk_matrix = np.fft.irfft(inverse_temporal_spectrum, n=input_matrix.shape[1], axis=1).real.astype(np.float32)

    normalized_mode = str(fk_normalize_mode).lower()
    if normalized_mode == "zscore":
        fk_mean = float(np.nanmean(fk_matrix))
        fk_std = float(np.nanstd(fk_matrix))
        if np.isfinite(fk_std) and fk_std > 1e-12:
            fk_matrix = (fk_matrix - fk_mean) / fk_std
    elif normalized_mode == "percentile":
        fk_low, fk_high = fast_percentile_sample(fk_matrix, (1, 99))
        fk_scale = max(abs(fk_low), abs(fk_high))
        if np.isfinite(fk_scale) and fk_scale > 1e-12:
            fk_matrix = fk_matrix / fk_scale
    return np.asarray(fk_matrix, dtype=np.float32)


def compute_fbe_rms_maps(fbe_accumulators, fbe_frequency_bands, fbe_colorbar_range):
    fbe_data_by_band = [fbe_accumulator.concat() for fbe_accumulator in fbe_accumulators]
    has_manual_colorbar_range = isinstance(fbe_colorbar_range, (tuple, list)) and len(fbe_colorbar_range) == 2
    if has_manual_colorbar_range:
        global_colorbar_min, global_colorbar_max = fbe_colorbar_range
    else:
        global_colorbar_min, global_colorbar_max = float("inf"), float("-inf")
        for band_matrix in fbe_data_by_band:
            if band_matrix is not None:
                band_min, band_max = fast_percentile_sample(band_matrix, (2, 98))
                global_colorbar_min = min(global_colorbar_min, band_min)
                global_colorbar_max = max(global_colorbar_max, band_max)
        if global_colorbar_min == float("inf"):
            global_colorbar_min, global_colorbar_max = 0, 1
    return {
        "fbe_band_matrices": fbe_data_by_band,
        "fbe_frequency_bands": fbe_frequency_bands,
        "global_colorbar_vmin": global_colorbar_min,
        "global_colorbar_vmax": global_colorbar_max,
    }


def _safe_band_limits(sample_rate_hz: float, low_cut_hz: float, high_cut_hz: float) -> tuple[float, float] | tuple[None, None]:
    nyquist_hz = 0.5 * float(sample_rate_hz)
    safe_low_hz = max(1e-6, float(low_cut_hz))
    safe_high_hz = min(float(high_cut_hz), nyquist_hz * 0.9)
    if safe_low_hz >= safe_high_hz:
        return None, None
    return safe_low_hz, safe_high_hz


def _safe_low_cut(sample_rate_hz: float, lowpass_cut_hz: float) -> float | None:
    nyquist_hz = 0.5 * float(sample_rate_hz)
    safe_cut_hz = max(1e-6, min(float(lowpass_cut_hz), nyquist_hz * 0.9))
    if safe_cut_hz >= nyquist_hz:
        return None
    return safe_cut_hz


def compute_lfdas_batch(
    input_data_matrix: np.ndarray,
    sample_rate_hz: float,
    lfdas_method: str = "bandpass",
    lfdas_high_hz: float = 0.01,
    lfdas_low_ratio: float = 0.005,
    filter_order: int = 4,
    use_zero_phase: bool = False,
) -> np.ndarray:
    """Batch path: compute LFDAS result from full-window data."""
    input_data_matrix = np.asarray(input_data_matrix, dtype=np.float32)
    if input_data_matrix.ndim != 2 or input_data_matrix.shape[1] == 0:
        return input_data_matrix.copy()
    if lfdas_method == "bandpass":
        low_cut_hz = float(lfdas_high_hz) * float(lfdas_low_ratio)
        safe_low_hz, safe_high_hz = _safe_band_limits(sample_rate_hz, low_cut_hz, lfdas_high_hz)
        if safe_low_hz is None:
            return input_data_matrix.copy()
        filtered_data_matrix = gpu_bandpass(
            input_data_matrix,
            float(sample_rate_hz),
            safe_low_hz,
            safe_high_hz,
            filter_order=int(filter_order),
            use_zero_phase=bool(use_zero_phase),
        )
        return _to_numpy_f32(filtered_data_matrix)
    safe_lowpass_cut_hz = _safe_low_cut(sample_rate_hz, lfdas_high_hz)
    if safe_lowpass_cut_hz is None:
        return input_data_matrix.copy()
    filtered_data_matrix = gpu_lowpass(
        input_data_matrix,
        float(sample_rate_hz),
        safe_lowpass_cut_hz,
        filter_order=int(filter_order),
        use_zero_phase=bool(use_zero_phase),
    )
    return _to_numpy_f32(filtered_data_matrix)


def compute_fbe_batch(
    input_data_matrix: np.ndarray,
    sample_rate_hz: float,
    frequency_bands: list[tuple[float, float]],
    smoothing_window_seconds: float = 0.1,
    filter_order: int = 4,
) -> list[np.ndarray | None]:
    """Batch path: compute FBE RMS maps for each band on full-window data."""
    input_data_matrix = np.asarray(input_data_matrix, dtype=np.float32)
    if input_data_matrix.ndim != 2 or input_data_matrix.shape[1] == 0:
        return [None for _ in frequency_bands]
    band_rms_results: list[np.ndarray | None] = []
    for band_low_hz, band_high_hz in frequency_bands:
        safe_low_hz, safe_high_hz = _safe_band_limits(sample_rate_hz, band_low_hz, band_high_hz)
        if safe_low_hz is None:
            band_rms_results.append(None)
            continue
        rms_matrix = gpu_band_rms(
            input_data_matrix,
            sample_rate_hz=float(sample_rate_hz),
            low_cut_hz=safe_low_hz,
            high_cut_hz=safe_high_hz,
            smoothing_window_seconds=smoothing_window_seconds,
            filter_order=int(filter_order),
        )
        band_rms_results.append(np.asarray(rms_matrix, dtype=np.float32, copy=False))
    return band_rms_results


def compute_lfdas_streaming_chunk(
    extended_chunk_matrix,
    sample_rate_hz: float,
    lfdas_method: str,
    lfdas_high_hz: float,
    lfdas_low_ratio: float,
    filter_order: int,
    use_zero_phase: bool,
    valid_left_padding_columns: int,
    valid_chunk_samples: int,
):
    """Streaming path: filter one extended chunk and return center region."""
    if lfdas_method == "bandpass":
        low_cut_hz = float(lfdas_high_hz) * float(lfdas_low_ratio)
        high_cut_hz = float(lfdas_high_hz)
        filtered_extended_matrix = gpu_bandpass(
            extended_chunk_matrix,
            sample_rate_hz,
            low_cut_hz,
            high_cut_hz,
            filter_order=filter_order,
            use_zero_phase=use_zero_phase,
        )
    else:
        filtered_extended_matrix = gpu_lowpass(
            extended_chunk_matrix,
            sample_rate_hz,
            float(lfdas_high_hz),
            filter_order=filter_order,
            use_zero_phase=use_zero_phase,
        )
    return filtered_extended_matrix[:, valid_left_padding_columns:valid_left_padding_columns + valid_chunk_samples]


def compute_fbe_streaming_chunk(
    streaming_chunk_matrix,
    sample_rate_hz: float,
    fbe_runtime_specs,
    smoothing_window_seconds: float,
    filter_order: int,
):
    """Streaming path: compute one chunk FBE RMS list by runtime specs."""
    band_rms_results = []
    for runtime_spec in fbe_runtime_specs:
        if not runtime_spec["is_enabled"]:
            band_rms_results.append(None)
            continue
        rms_matrix = gpu_band_rms(
            streaming_chunk_matrix,
            sample_rate_hz,
            runtime_spec["runtime_low_hz"],
            runtime_spec["runtime_high_hz"],
            smoothing_window_seconds=smoothing_window_seconds,
            filter_order=filter_order,
        )
        band_rms_results.append(rms_matrix)
    return band_rms_results

