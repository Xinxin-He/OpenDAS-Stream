from __future__ import annotations

import math
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import MultipleLocator, ScalarFormatter, MaxNLocator, FuncFormatter

import compute_core as cc

TITLE_MAIN_FONT_SIZE = 11
TITLE_SUB_FONT_SIZE = 9
SUPTITLE_FONT_SIZE = 13
COLORBAR_RELATIVE_X = 1.008
COLORBAR_RELATIVE_WIDTH = 0.022
COLORBAR_RELATIVE_Y = 0.0
COLORBAR_RELATIVE_HEIGHT = 1.0
COLORBAR_TICK_FONT_SIZE = 8
COLORBAR_LABEL_FONT_SIZE = 8
SUPTITLE_PAD = 1.04


def _build_channel_formatter(plot_context):
    dx_m = float(plot_context["dx"])
    channel_inside = int(plot_context["ch_inside"])
    channel_start = int(plot_context["ch_start"])
    channel_label_mode = plot_context.get("channel_label_mode", "absolute")
    if dx_m <= 0:
        return FuncFormatter(lambda _, __: "")

    def _to_channel_label(y_value, _pos):
        absolute_channel_index = int(round(channel_inside + (y_value / dx_m)))
        if channel_label_mode == "relative_to_ch_start":
            return f"{absolute_channel_index - channel_start}"
        return f"{absolute_channel_index}"

    return FuncFormatter(_to_channel_label)


def apply_axis_style(axis, plot_context, axis_kind="heatmap"):
    if axis_kind != "freq_dist":
        if plot_context["x_axis_mode"] == "datetime":
            axis.xaxis_date()
            locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            formatter = mdates.ConciseDateFormatter(locator)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)
            axis.set_xlabel("Time (Abs)")
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
            axis.set_xlabel("Time (s)")

    if axis_kind not in ("heatmap", "freq_dist"):
        return

    if plot_context["y_mode"] == "channel":
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, min_n_ticks=4))
        axis.yaxis.set_major_formatter(_build_channel_formatter(plot_context))
        if plot_context.get("channel_label_mode", "absolute") == "relative_to_ch_start":
            axis.set_ylabel("Channel (Relative to CH_START)")
        else:
            axis.set_ylabel("Channel (Absolute)")
    else:
        axis.yaxis.set_major_locator(MultipleLocator(plot_context["y_tick_interval"]))
        axis.set_ylabel("Distance (m)")


def _add_colorbar(image_artist, axis, label=None):
    colorbar_axis = axis.inset_axes(
        [
            COLORBAR_RELATIVE_X,
            COLORBAR_RELATIVE_Y,
            COLORBAR_RELATIVE_WIDTH,
            COLORBAR_RELATIVE_HEIGHT,
        ],
        transform=axis.transAxes,
    )
    colorbar = axis.figure.colorbar(image_artist, cax=colorbar_axis)
    colorbar_formatter = ScalarFormatter(useMathText=True)
    colorbar_formatter.set_scientific(True)
    colorbar_formatter.set_powerlimits((-3, 3))
    colorbar.formatter = colorbar_formatter
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONT_SIZE)
    colorbar.ax.yaxis.get_offset_text().set_size(COLORBAR_TICK_FONT_SIZE)
    if label is not None:
        colorbar.set_label(label, fontsize=COLORBAR_LABEL_FONT_SIZE, labelpad=4)
    return colorbar


def _draw_raw_vis(axis, plot_context):
    raw_heatmap_info = cc.compute_raw_heatmap(
        plot_context["res_raw"],
        plot_context["do_common_mode"],
    )
    image_artist = axis.imshow(
        raw_heatmap_info["visual_matrix"],
        aspect="auto",
        cmap=raw_heatmap_info["colormap_name"],
        vmin=raw_heatmap_info["colorbar_vmin"],
        vmax=raw_heatmap_info["colorbar_vmax"],
        extent=plot_context["extent"],
        interpolation="nearest",
        rasterized=True,
    )
    common_mode_text = (
        f"on ({plot_context['common_mode_method']})"
        if plot_context["do_common_mode"]
        else "off"
    )
    demean_text = "on" if plot_context["do_raw_per_channel_demean"] else "off"
    axis.set_title(
        "Raw DAS Heatmap\n"
        f"(Common mode: {common_mode_text}, Per-channel demean: {demean_text})",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    _add_colorbar(image_artist, axis)
    apply_axis_style(axis, plot_context, axis_kind="heatmap")


def _draw_freq_dist(axis, plot_context):
    frequency_distance_info = cc.compute_freq_distance_spectrum(
        plot_context["res_raw"],
        plot_context["visual_sample_rate_hz"],
        plot_context["y_max"],
        plot_context["y_min"],
    )
    image_artist = axis.imshow(
        frequency_distance_info["spectrum_matrix"],
        aspect="auto",
        cmap="jet",
        vmin=frequency_distance_info["colorbar_vmin"],
        vmax=frequency_distance_info["colorbar_vmax"],
        extent=frequency_distance_info["plot_extent"],
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(
        "Frequency-Distance Spectrum\n"
        f"(Visual sampling rate: {plot_context['visual_sample_rate_hz']:.3g} Hz)",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    axis.set_xlabel(frequency_distance_info["x_axis_label"])
    _add_colorbar(image_artist, axis, label="Magnitude")
    apply_axis_style(axis, plot_context, axis_kind="freq_dist")


def _draw_band_map(axis, plot_context, band_matrix, band_frequency_pair, plot_title, is_envelope_band):
    band_map_info = cc.compute_bandpass_map(
        band_matrix,
        band_frequency_pair,
        is_envelope_band,
    )
    image_artist = axis.imshow(
        band_map_info["visual_matrix"],
        aspect="auto",
        cmap=band_map_info["colormap_name"],
        vmin=band_map_info["colorbar_vmin"],
        vmax=band_map_info["colorbar_vmax"],
        extent=plot_context["extent"],
        interpolation="nearest",
        rasterized=True,
    )
    low_hz, high_hz = band_frequency_pair
    axis.set_title(
        f"{plot_title}\n"
        f"(Band: {low_hz:g}-{high_hz:g} Hz, Order: {plot_context['filter_order']}, "
        f"Zero-phase: {plot_context['use_zero_phase']})",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    _add_colorbar(image_artist, axis)
    apply_axis_style(axis, plot_context, axis_kind="heatmap")


def _draw_lfdas(axis, plot_context):
    lfdas_matrix = plot_context["acc_lfdas"].concat()
    lfdas_map_info = cc.compute_lfdas_map(
        lfdas_matrix,
        plot_context["lfdas_method"],
        plot_context["lfdas_high_hz"],
        plot_context["lfdas_low_ratio"],
    )
    image_artist = axis.imshow(
        lfdas_map_info["visual_matrix"],
        aspect="auto",
        cmap=lfdas_map_info["colormap_name"],
        vmin=lfdas_map_info["colorbar_vmin"],
        vmax=lfdas_map_info["colorbar_vmax"],
        extent=plot_context["extent"],
        interpolation="nearest",
        rasterized=True,
    )
    axis.set_title(
        "Low-Frequency DAS Map\n"
        f"({lfdas_map_info['plot_title']}, Order: {plot_context['filter_order']}, "
        f"Zero-phase: {plot_context['use_zero_phase']})",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    _add_colorbar(image_artist, axis)
    apply_axis_style(axis, plot_context, axis_kind="heatmap")


def _resolve_target_channel_row(plot_context, target_channel_index):
    if target_channel_index is None:
        target_channel_row_index = plot_context["final_ch"] // 2
        display_channel = int(plot_context["raw_ch_indices"][target_channel_row_index])
        return target_channel_row_index, display_channel
    if (
        target_channel_index < plot_context["ch_start"]
        or target_channel_index >= plot_context["ch_end"]
    ):
        target_channel_row_index = 0 if target_channel_index < plot_context["ch_start"] else plot_context["final_ch"] - 1
        return target_channel_row_index, target_channel_index
    target_channel_row_index = int(
        (target_channel_index - plot_context["ch_start"]) / plot_context["r_ch"]
    )
    target_channel_row_index = max(0, min(target_channel_row_index, plot_context["final_ch"] - 1))
    display_channel = int(plot_context["raw_ch_indices"][target_channel_row_index])
    return target_channel_row_index, display_channel


def _draw_spectrogram(axis, plot_context, target_channel_index):
    target_channel_row_index, display_channel = _resolve_target_channel_row(plot_context, target_channel_index)
    window_sec = plot_context.get("spectrogram_window_sec")
    overlap_ratio = float(plot_context.get("spectrogram_overlap_ratio", 0.5))
    base_nperseg = int(plot_context.get("spectrogram_nperseg", 256))
    trace_len = int(plot_context["res_raw"].shape[1])
    if window_sec is None:
        nperseg = max(16, min(base_nperseg, trace_len))
    else:
        nperseg = int(round(float(window_sec) * float(plot_context["visual_sample_rate_hz"])))
        nperseg = max(16, min(nperseg, trace_len))
    noverlap = int(round(overlap_ratio * nperseg))
    noverlap = max(0, min(noverlap, nperseg - 1))
    spectrogram_info = cc.compute_channel_spectrogram(
        plot_context["res_raw"],
        plot_context["visual_sample_rate_hz"],
        target_channel_row_index,
        plot_context["start_dt_global"],
        plot_context["x_axis_mode"],
        plot_context["t_start"],
        spectrogram_nperseg=nperseg,
        spectrogram_noverlap=noverlap,
    )
    real_depth_m = (display_channel - plot_context["ch_inside"]) * plot_context["dx"]
    actual_window_sec = nperseg / float(plot_context["visual_sample_rate_hz"]) if plot_context["visual_sample_rate_hz"] > 0 else 0.0
    if plot_context["x_axis_mode"] == "datetime":
        datetime_axis = [mdates.date2num(time_value) for time_value in spectrogram_info["time_axis"]]
        image_artist = axis.pcolormesh(
            datetime_axis,
            spectrogram_info["frequency_axis_hz"],
            spectrogram_info["spectrogram_log_power"],
            shading="gouraud",
            cmap="inferno",
        )
    else:
        image_artist = axis.pcolormesh(
            spectrogram_info["time_axis"],
            spectrogram_info["frequency_axis_hz"],
            spectrogram_info["spectrogram_log_power"],
            shading="gouraud",
            cmap="inferno",
        )
    axis.set_title(
        "Channel Spectrogram\n"
        f"(Channel: {display_channel}, Depth: {real_depth_m:.1f} m, "
        f"Window: {actual_window_sec:.3g}s, Overlap: {overlap_ratio:.2f}, "
        f"Visual Fs: {plot_context['visual_sample_rate_hz']:.3g} Hz)",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    axis.set_ylabel("Frequency (Hz)")
    _add_colorbar(image_artist, axis, label="dB")
    apply_axis_style(axis, plot_context, axis_kind="spectrogram")


def _draw_fk(axis, plot_context):
    fk_colorbar_range = plot_context.get("fk_colorbar_range", "auto")
    use_manual_range = isinstance(fk_colorbar_range, (tuple, list)) and len(fk_colorbar_range) == 2
    if use_manual_range:
        colorbar_vmin = float(fk_colorbar_range[0])
        colorbar_vmax = float(fk_colorbar_range[1])
    else:
        raw_low, raw_high = cc.fast_percentile_sample(plot_context["res_raw"], (1, 99))
        fk_low, fk_high = cc.fast_percentile_sample(plot_context["res_fk"], (1, 99))
        symmetric_limit = max(abs(raw_low), abs(raw_high), abs(fk_low), abs(fk_high))
        colorbar_vmin, colorbar_vmax = -symmetric_limit, symmetric_limit
    image_artist = axis.imshow(
        plot_context["res_fk"],
        aspect="auto",
        cmap="seismic",
        vmin=colorbar_vmin,
        vmax=colorbar_vmax,
        extent=plot_context["extent"],
        interpolation="nearest",
        rasterized=True,
    )
    fk_direction_text = {
        "positive": "k>=0",
        "negative": "k<=0",
        "both": "k all",
    }.get(str(plot_context.get("fk_pass_direction", "positive")).lower(), "k>=0")
    fk_prefilter_band_hz = plot_context.get("fk_prefilter_band_hz")
    if fk_prefilter_band_hz is None:
        fk_prefilter_text = "prefilter: off"
    else:
        fk_prefilter_text = f"prefilter: {fk_prefilter_band_hz[0]:g}-{fk_prefilter_band_hz[1]:g} Hz"
    fk_normalize_text = str(plot_context.get("fk_normalize_mode", "none")).lower()
    axis.set_title(
        "F-K Comparison Map\n"
        f"({fk_direction_text}, {fk_prefilter_text}, norm: {fk_normalize_text}, "
        f"dx: {plot_context['dx'] * plot_context['r_ch']:.3g} m)",
        fontsize=TITLE_SUB_FONT_SIZE,
        pad=6,
    )
    _add_colorbar(image_artist, axis)
    apply_axis_style(axis, plot_context, axis_kind="heatmap")


def draw_figure_one_band_layout(plot_context, band_accumulators, frequency_bands, env_matrix, env_band):
    band_count = len(frequency_bands)
    figure = plt.figure(figsize=(12, max(6, 2.2 * band_count + 2.0)), constrained_layout=True)
    root_grid = figure.add_gridspec(2, 2, height_ratios=[1.0, max(1.0, float(band_count))])
    raw_axis = figure.add_subplot(root_grid[0, 0])
    freq_axis = figure.add_subplot(root_grid[0, 1])
    super_title = figure.suptitle(
        "Figure 1 - Raw, Frequency-Distance, Bandpass and Envelope",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    super_title.set_y(SUPTITLE_PAD)
    _draw_raw_vis(raw_axis, plot_context)
    _draw_freq_dist(freq_axis, plot_context)

    left_band_grid = root_grid[1, 0].subgridspec(band_count, 1)
    first_band_axis = None
    for band_index, band_frequency_pair in enumerate(frequency_bands):
        if first_band_axis is None:
            band_axis = figure.add_subplot(left_band_grid[band_index, 0], sharex=raw_axis, sharey=raw_axis)
            first_band_axis = band_axis
        else:
            band_axis = figure.add_subplot(left_band_grid[band_index, 0], sharex=first_band_axis, sharey=first_band_axis)
        band_matrix = band_accumulators[band_index].concat()
        if band_matrix is None:
            band_axis.text(0.5, 0.5, f"BANDPLOTS {band_index + 1} unavailable", ha="center", va="center", transform=band_axis.transAxes)
            continue
        _draw_band_map(
            band_axis,
            plot_context,
            band_matrix,
            band_frequency_pair,
            f"Bandpass Map {band_index + 1}",
            is_envelope_band=False,
        )

    env_axis = figure.add_subplot(root_grid[1, 1], sharex=raw_axis, sharey=raw_axis)
    _draw_band_map(
        env_axis,
        plot_context,
        env_matrix,
        env_band,
        "Band Envelope Map",
        is_envelope_band=True,
    )


def draw_figure_two_fbe_layout(
    plot_context,
    fbe_accumulators,
    fbe_frequency_bands,
    fbe_colorbar_range,
    fbe_runtime_specs,
    fbe_rms_window_seconds,
):
    fbe_plot_info = cc.compute_fbe_rms_maps(
        fbe_accumulators,
        fbe_frequency_bands,
        fbe_colorbar_range,
    )
    fbe_data_by_band = fbe_plot_info["fbe_band_matrices"]
    global_colorbar_min = fbe_plot_info["global_colorbar_vmin"]
    global_colorbar_max = fbe_plot_info["global_colorbar_vmax"]

    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    root_grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 2.0])
    super_title = figure.suptitle(
        "Figure 2 - Raw, Frequency-Distance and FBE-RMS",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    super_title.set_y(SUPTITLE_PAD)
    raw_axis = figure.add_subplot(root_grid[0, 0])
    freq_axis = figure.add_subplot(root_grid[0, 1])
    _draw_raw_vis(raw_axis, plot_context)
    _draw_freq_dist(freq_axis, plot_context)

    fbe_grid = root_grid[1, :].subgridspec(2, 2)
    for fbe_index in range(4):
        fbe_axis = figure.add_subplot(
            fbe_grid[fbe_index // 2, fbe_index % 2],
            sharex=raw_axis,
            sharey=raw_axis,
        )
        if fbe_index >= len(fbe_data_by_band) or fbe_index >= len(fbe_frequency_bands):
            fbe_axis.text(0.5, 0.5, f"FBE {fbe_index + 1} unavailable", ha="center", va="center", transform=fbe_axis.transAxes)
            continue
        band_low_hz, band_high_hz = fbe_frequency_bands[fbe_index]
        band_matrix = fbe_data_by_band[fbe_index]
        runtime_spec = fbe_runtime_specs[fbe_index] if fbe_index < len(fbe_runtime_specs) else None
        if band_matrix is None:
            unavailable_note = runtime_spec["note_text"] if runtime_spec and runtime_spec.get("note_text") else f"FBE {fbe_index + 1} unavailable"
            fbe_axis.text(0.5, 0.5, unavailable_note, ha="center", va="center", transform=fbe_axis.transAxes)
            continue
        image_artist = fbe_axis.imshow(
            band_matrix,
            aspect="auto",
            cmap="jet",
            extent=plot_context["extent"],
            vmin=global_colorbar_min,
            vmax=global_colorbar_max,
            interpolation="nearest",
            rasterized=True,
        )
        is_clipped = (
            runtime_spec
            and runtime_spec["is_enabled"]
            and abs(runtime_spec["runtime_high_hz"] - runtime_spec["requested_high_hz"]) > 1e-12
        )
        clipped_suffix = ", clipped" if is_clipped else ""
        fbe_axis.set_title(
            f"FBE RMS Map {fbe_index + 1}\n"
            f"(Band: {band_low_hz:g}-{band_high_hz:g} Hz, "
            f"RMS window: {fbe_rms_window_seconds:g}s{clipped_suffix})",
            fontsize=TITLE_SUB_FONT_SIZE,
            pad=6,
        )
        _add_colorbar(image_artist, fbe_axis, label="RMS")
        apply_axis_style(fbe_axis, plot_context, axis_kind="heatmap")
    if len(fbe_frequency_bands) != 4:
        print(f"[Figure 2] Layout uses first 4 FBE panels; configured bands={len(fbe_frequency_bands)}.")


def draw_figure_three_lfdas_layout(plot_context):
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    root_grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 2.0])
    super_title = figure.suptitle(
        "Figure 3 - Raw, Frequency-Distance and LFDAS",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    super_title.set_y(SUPTITLE_PAD)
    raw_axis = figure.add_subplot(root_grid[0, 0])
    freq_axis = figure.add_subplot(root_grid[0, 1])
    lfdas_axis = figure.add_subplot(root_grid[1, :], sharex=raw_axis, sharey=raw_axis)
    _draw_raw_vis(raw_axis, plot_context)
    _draw_freq_dist(freq_axis, plot_context)
    _draw_lfdas(lfdas_axis, plot_context)


def draw_figure_four_fk_layout(plot_context):
    figure, fk_axis = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    super_title = figure.suptitle(
        "Figure 4 - F-K Comparison",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    super_title.set_y(SUPTITLE_PAD)
    _draw_fk(fk_axis, plot_context)


def draw_figure_five_spectrogram_stack(plot_context, target_channel_indices):
    if not target_channel_indices:
        return
    plot_count = len(target_channel_indices)
    figure, axes = plt.subplots(
        plot_count,
        1,
        figsize=(12, max(4.0, plot_count * 3.2)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    super_title = figure.suptitle(
        "Figure 5 - Channel Spectrogram Stack",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight="bold",
    )
    super_title.set_y(SUPTITLE_PAD)
    axes = np.atleast_1d(axes)
    for plot_index, target_channel_index in enumerate(target_channel_indices):
        _draw_spectrogram(axes[plot_index], plot_context, target_channel_index)


def build_plot_context(
    *,
    plot_build_config,
):
    acc_raw = plot_build_config["acc_raw"]
    acc_bands = plot_build_config["acc_bands"]
    acc_lfdas = plot_build_config["acc_lfdas"]
    do_fk = plot_build_config["do_fk"]
    manual_dx_m = plot_build_config["manual_dx_m"]
    channel_downsample_ratio = plot_build_config["channel_downsample_ratio"]
    time_downsample_ratio = plot_build_config["time_downsample_ratio"]
    sample_rate_hz = plot_build_config["sample_rate_hz"]
    ch_start = plot_build_config["ch_start"]
    ch_end = plot_build_config["ch_end"]
    ch_inside = plot_build_config["ch_inside"]
    do_common_mode = plot_build_config["do_common_mode"]
    common_mode_method = plot_build_config["common_mode_method"]
    do_raw_per_channel_demean = plot_build_config["do_raw_per_channel_demean"]
    lfdas_method = plot_build_config["lfdas_method"]
    lfdas_high_hz = plot_build_config["lfdas_high_hz"]
    lfdas_low_ratio = plot_build_config["lfdas_low_ratio"]
    filter_order = plot_build_config["filter_order"]
    use_zero_phase = plot_build_config["use_zero_phase"]
    x_axis_mode = plot_build_config["x_axis_mode"]
    y_mode = plot_build_config["y_mode"]
    channel_label_mode = plot_build_config["channel_label_mode"]
    y_tick_interval = plot_build_config["y_tick_interval"]
    time_axis_gap_mode = plot_build_config["time_axis_gap_mode"]
    real_span_seconds = plot_build_config["real_span_seconds"]
    file_infos = plot_build_config["file_infos"]
    start_dt_global = plot_build_config["start_dt_global"]
    t_start = plot_build_config["t_start"]
    fk_pass_direction = plot_build_config["fk_pass_direction"]
    fk_prefilter_band_hz = plot_build_config["fk_prefilter_band_hz"]
    fk_normalize_mode = plot_build_config["fk_normalize_mode"]
    fk_colorbar_range = plot_build_config["fk_colorbar_range"]
    spectrogram_window_sec = plot_build_config["spectrogram_window_sec"]
    spectrogram_nperseg = plot_build_config["spectrogram_nperseg"]
    spectrogram_overlap_ratio = plot_build_config["spectrogram_overlap_ratio"]
    visual_sample_rate_hz = sample_rate_hz / time_downsample_ratio

    res_raw = acc_raw.concat() if acc_raw else None
    res_fk = None
    if do_fk and res_raw is not None:
        dx_for_fk = manual_dx_m * channel_downsample_ratio
        res_fk = cc.compute_fk_compare(
            res_raw,
            dx_for_fk,
            fk_pass_direction=fk_pass_direction,
            fk_prefilter_band_hz=fk_prefilter_band_hz,
            sample_rate_hz=visual_sample_rate_hz,
            filter_order=filter_order,
            fk_normalize_mode=fk_normalize_mode,
        )
        if res_fk is not None:
            print(f"[F-K] Done: input {res_raw.shape}, output {res_fk.shape}")
        else:
            print("[F-K] Skipped: invalid dx or FK prefilter settings.")

    ref_matrix = res_raw if res_raw is not None else (acc_bands[0].concat() if acc_bands else None)
    if ref_matrix is None:
        print("[Error] No data was processed.")
        return None

    final_channel_count, final_time_count = ref_matrix.shape
    dx_m = manual_dx_m
    raw_channel_indices = np.arange(final_channel_count) * channel_downsample_ratio + ch_start
    y_values_m = (raw_channel_indices - ch_inside) * dx_m
    y_min_m, y_max_m = y_values_m[0], y_values_m[-1]
    visible_duration_seconds = final_time_count / visual_sample_rate_hz

    if time_axis_gap_mode == "realtime" and real_span_seconds is not None and file_infos:
        if x_axis_mode == "datetime":
            first_file_dt = file_infos[0][2]
            d_start_num = mdates.date2num(first_file_dt)
            d_end_num = mdates.date2num(first_file_dt + timedelta(seconds=real_span_seconds))
            extent = [d_start_num, d_end_num, y_max_m, y_min_m]
        else:
            extent = [0, real_span_seconds, y_max_m, y_min_m]
    else:
        if x_axis_mode == "datetime":
            d_start_num = mdates.date2num(start_dt_global)
            d_end_num = mdates.date2num(start_dt_global + timedelta(seconds=visible_duration_seconds))
            extent = [d_start_num, d_end_num, y_max_m, y_min_m]
        else:
            extent = [t_start, t_start + visible_duration_seconds, y_max_m, y_min_m]

    return {
        "res_raw": res_raw,
        "res_fk": res_fk,
        "acc_lfdas": acc_lfdas,
        "acc_bands": acc_bands,
        "extent": extent,
        "do_common_mode": do_common_mode,
        "common_mode_method": common_mode_method,
        "do_raw_per_channel_demean": do_raw_per_channel_demean,
        "lfdas_method": lfdas_method,
        "lfdas_high_hz": lfdas_high_hz,
        "lfdas_low_ratio": lfdas_low_ratio,
        "filter_order": filter_order,
        "use_zero_phase": use_zero_phase,
        "x_axis_mode": x_axis_mode,
        "y_mode": y_mode,
        "channel_label_mode": channel_label_mode,
        "start_dt_global": start_dt_global,
        "t_start": t_start,
        "visual_sample_rate_hz": visual_sample_rate_hz,
        "raw_ch_indices": raw_channel_indices,
        "final_ch": final_channel_count,
        "r_ch": channel_downsample_ratio,
        "ch_start": ch_start,
        "ch_end": ch_end,
        "ch_inside": ch_inside,
        "dx": dx_m,
        "y_max": y_max_m,
        "y_min": y_min_m,
        "y_tick_interval": y_tick_interval,
        "fk_pass_direction": fk_pass_direction,
        "fk_prefilter_band_hz": fk_prefilter_band_hz,
        "fk_normalize_mode": fk_normalize_mode,
        "fk_colorbar_range": fk_colorbar_range,
        "spectrogram_window_sec": spectrogram_window_sec,
        "spectrogram_nperseg": spectrogram_nperseg,
        "spectrogram_overlap_ratio": spectrogram_overlap_ratio,
    }


def render_enabled_figures(
    *,
    plot_context,
    render_config,
):
    if plot_context is None:
        return

    do_bandplots = render_config["do_bandplots"]
    do_fbe = render_config["do_fbe"]
    do_lfdas = render_config["do_lfdas"]
    do_fk = render_config["do_fk"]
    do_spectrogram = render_config["do_spectrogram"]
    acc_bands = render_config["acc_bands"]
    bands = render_config["bands"]
    acc_band_env = render_config["acc_band_env"]
    bandplots_env_band = render_config["bandplots_env_band"]
    acc_fbe = render_config["acc_fbe"]
    fbe_bands = render_config["fbe_bands"]
    fbe_colorbar_range = render_config["fbe_colorbar_range"]
    fbe_specs = render_config["fbe_specs"]
    fbe_rms_window_sec = render_config["fbe_rms_window_sec"]
    spectrogram_target_channels = render_config["spectrogram_target_channels"]

    res_raw = plot_context["res_raw"]
    res_fk = plot_context["res_fk"]

    # 图一：RAW_VIS + FREQ_DIST + BANDPLOTS + BANDPLOTS ENV
    if do_bandplots:
        env_matrix = acc_band_env.concat() if acc_band_env is not None else None
        if res_raw is None:
            print("[Figure 1] Skipped: RAW_VIS source matrix is unavailable.")
        elif env_matrix is None:
            print("[Figure 1] Skipped: BANDPLOTS ENV matrix is unavailable.")
        else:
            draw_figure_one_band_layout(
                plot_context=plot_context,
                band_accumulators=acc_bands,
                frequency_bands=bands,
                env_matrix=env_matrix,
                env_band=bandplots_env_band,
            )

    # 图二：RAW_VIS + FREQ_DIST + 4 个 FBE
    if do_fbe and res_raw is not None:
        draw_figure_two_fbe_layout(
            plot_context=plot_context,
            fbe_accumulators=acc_fbe,
            fbe_frequency_bands=fbe_bands,
            fbe_colorbar_range=fbe_colorbar_range,
            fbe_runtime_specs=fbe_specs,
            fbe_rms_window_seconds=fbe_rms_window_sec,
        )

    # 图三：RAW_VIS + FREQ_DIST + LFDAS
    if do_lfdas:
        if plot_context["acc_lfdas"] is None or plot_context["acc_lfdas"].concat() is None:
            print("[Figure 3] Skipped: LFDAS matrix is unavailable.")
        elif res_raw is None:
            print("[Figure 3] Skipped: RAW_VIS source matrix is unavailable.")
        else:
            draw_figure_three_lfdas_layout(plot_context)

    # 图四：FK
    if do_fk and res_fk is not None:
        draw_figure_four_fk_layout(plot_context)

    # 图五：若干 DO_SPECTROGRAM（单列纵向）
    if do_spectrogram and res_raw is not None:
        target_channel_indices = (
            spectrogram_target_channels
            if isinstance(spectrogram_target_channels, list)
            else [spectrogram_target_channels]
        )
        draw_figure_five_spectrogram_stack(
            plot_context=plot_context,
            target_channel_indices=target_channel_indices,
        )
    plt.show()
