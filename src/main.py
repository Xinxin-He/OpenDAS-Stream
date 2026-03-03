# -*- coding: utf-8 -*-

from pathlib import Path
import re
import h5py
from datetime import datetime, timedelta
from time import perf_counter
import math
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from tqdm import tqdm
import gc
import compute_core as cc
import plotters as pl
from filter_core import (
    HAS_GPU,
    HAS_TDMS,
    cp,
    cpx_sig,
    TdmsFile,
    get_xp,
    remove_common_mode,
    gpu_downsample,
    gpu_bandpass,
    gpu_lowpass,
    gpu_highpass,
    gpu_envelope,
    gpu_band_rms,
)

# =================================================================
#                             GPU检查
#           GPU/CPU 后端由 filter_core 统一初始化与管理
# =================================================================


# =================================================================
#                  用户配置区域 (User Configuration)
# =================================================================

# -----------------------------------------------------------------
# A. 运行入口与数据范围（最高优先级）
# -----------------------------------------------------------------
# 用途：先确定“读哪里、读哪段时间”，这是整个任务能否起跑的第一条件。
# 说明：这里通常是前端最先要给后端的参数。
# --- [路径设置] ---
# 本地脚本：自动获取路径；WebUI：建议由文件夹选择器回传绝对路径
SCRIPT_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
IN_DIR = SCRIPT_DIR.parent / "data"  # 输入数据文件夹路径

# --- [文件检索] ---
# 格式：YYYYMMDDHHMM（12位）
# 示例：202506022130 -> 2025-06-02 21:30
# 备选：如需秒级文件名匹配，扫描逻辑也支持 YYYYMMDD_HHMMSS(.sss) 样式文件名
START_TS = "202506020500"  # 检索开始时间（含）
END_TS   = "202506020507"  # 检索结束时间（含）

# -----------------------------------------------------------------
# B. 硬件与物理真值（必须正确）
# -----------------------------------------------------------------
# 用途：这些参数是“物理真值”（Source of Truth），直接影响频率轴、距离轴和通道校验。
# 原则：后端不再依赖文件元数据决定这些值；元数据仅用于校验与排障。
# --- [核心物理参数] ---
# MANUAL_FS_HZ:
#   单位：Hz
#   建议：与采集配置完全一致；错误会导致全部频域结果偏移
MANUAL_FS_HZ = 100  # 采样率 (Hz) - 必填，必须 > 0
# MANUAL_DX_M:
#   单位：m/通道
#   作用：影响距离轴、FK 中等效空间采样
MANUAL_DX_M = 1.0  # 通道间距 (m) - 必填，必须 > 0
# MANUAL_GAUGE_LENGTH_M:
#   单位：m
#   当前：本版主要用于参数校验，后续算法会逐步接入
MANUAL_GAUGE_LENGTH_M = 4  # 量距长度 (m) - 必填，必须 > 0
# MANUAL_N_CHANNELS_EXPECTED:
#   作用：用于读文件时的总通道数一致性校验
MANUAL_N_CHANNELS_EXPECTED = 1565  # 文件期望总通道数 - 必填，必须为正整数
# MANUAL_START_DISTANCE_M:
#   单位：m
#   作用：可作为绝对里程参考（当前主要用于前后端语义对齐）
MANUAL_START_DISTANCE_M = 0.0  # 绝对距离起点 (m)
# ON_CHANNEL_MISMATCH:
#   可选值：
#     - "error": 遇到通道数不一致立即报错终止
#     - "skip":  跳过异常文件并继续
#   默认："skip"
ON_CHANNEL_MISMATCH = "skip"

# 为兼容历史代码变量命名，内部统一映射为 fs。
fs = MANUAL_FS_HZ

# -----------------------------------------------------------------
# C. 时空裁剪（ROI）
# -----------------------------------------------------------------
# 用途：定义“从哪里看、看多长时间”，决定内存占用和计算规模。
# --- [空间裁剪 (Spatial Slice)] ---
# CH_START / CH_END:
#   含义：左闭右开区间 [CH_START, CH_END)
#   约束：0 <= CH_START < CH_END <= MANUAL_N_CHANNELS_EXPECTED
CH_START  = 400   # 起始通道号（含）
CH_END    = 1400  # 结束通道号（不含）
# CH_INSIDE:
#   含义：距离零点参考通道（用于相对距离/井口深度换算）
CH_INSIDE = 112

# --- [时间裁剪 (Temporal Slice)] ---
# 相对于所选文件序列起点的秒数（与 X_AXIS_MODE 无关）
t_start = 0     # 裁剪起始时间 (s)
t_end   = None  # 裁剪结束时间 (s); None 表示直到数据末尾

# --- [时间轴间隔模式] ---
# TIME_AXIS_GAP_MODE:
#   可选值：
#     - "realtime": 文件间时间空白保留为 NaN，钟表时间真实对齐
#     - "compact":  跳过空白后紧密拼接，适合连续浏览
TIME_AXIS_GAP_MODE = "realtime"

# -----------------------------------------------------------------
# D. 主任务开关与处理模式
# -----------------------------------------------------------------
# 用途：控制是否执行某类分析/绘图任务。
# 建议：先开核心图（RAW/LFDAS/FBE），再逐步打开其余任务。
# WebUI以下图做成勾选框
USE_STREAMING_PROCESSING = True  # True=流式；False=非流式批处理

DO_RAW_VIS     = True  # 原始数据热力图 (Waterfall)
DO_FREQ_DIST   = True  # 频率-距离谱 (F-D Spectrum)
DO_SPECTROGRAM = True  # 指定通道时频图
DO_LFDAS       = True  # 低频慢漂移图
DO_BANDPLOTS   = True  # 多频带滤波图 + ENV
DO_FK          = True  # F-K 对比图（需 dx 有效）
DO_FBE         = True  # 频带能量 RMS 热力图

# -----------------------------------------------------------------
# E. 各算法专属参数（按功能归并）
# -----------------------------------------------------------------
# 说明：本节只放算法参数，不再混放显示参数。

# --- [E1. LFDAS 参数] ---
# LFDAS_METHOD:
#   可选值：
#     - "bandpass": 推荐。更稳健抑制背景色块，适合事件识别
#     - "lowpass":  保留更强绝对趋势，适合趋势观察
LFDAS_METHOD = "bandpass"
# LFDAS_HIGH_HZ / LFDAS_LOW_RATIO:
#   当 method="lowpass" 时仅使用 LFDAS_HIGH_HZ
#   当 method="bandpass" 时：low_hz = LFDAS_HIGH_HZ * LFDAS_LOW_RATIO
LFDAS_HIGH_HZ = 0.01   # Hz，上限截止
LFDAS_LOW_RATIO = 0.05  # 比例系数（仅 bandpass 生效）
# LFDAS_OVERLAP_CYCLES:
#   作用：流式 chunk 过滤前的 overlap-save padding 周期数
#   典型候选：[2, 3, 4, 5]
#   调大：边界伪影更少，但内存/算力占用更高
#   调小：速度更快，但边界更容易出现瞬态伪影
LFDAS_OVERLAP_CYCLES = 5
# LFDAS_STREAM_STITCH_ENABLE / LFDAS_STREAM_STITCH_SEC:
#   作用：相邻 chunk 边界做 cross-fade，降低可视断层
#   仅影响拼接显示，不改 chunk 内部滤波结果
#   LFDAS_STREAM_STITCH_SEC 推荐：
#     - 3~5 s：细节更清晰（当前默认）
#     - 10~20 s：更平滑但可能偏“糊”
LFDAS_STREAM_STITCH_ENABLE = True
LFDAS_STREAM_STITCH_SEC = 3
# LFDAS_BATCH_PADDING_ENABLE:
#   作用：非流式（整窗）LFDAS 执行边界 padding，降低首尾滤波瞬态
#   建议：True（与流式边界补偿理念对齐）
LFDAS_BATCH_PADDING_ENABLE = True
# LFDAS_BATCH_PADDING_MODE:
#   当前可选值：
#     - "reflect": 首尾镜像反射 padding（推荐）
#   说明：首端和尾端都会对称 padding，避免只修一端导致另一端异常
LFDAS_BATCH_PADDING_MODE = "reflect"
# LFDAS_BATCH_PAD_MAX_SEC:
#   作用：限制非流式单侧 padding 的最长秒数，避免超低频时内存暴涨
#   建议：60~180s；越大边界更稳，但内存更高
LFDAS_BATCH_PAD_MAX_SEC = 120.0
# 说明（很重要）：
#   非流式 padding 的“最合理”默认策略是：只使用固定秒上限，不随窗口长度变化。
#   这样同一频带下，1分钟窗口与10分钟窗口的 padding 长度一致（除非窗口太短导致 reflect 受限）。

# --- [E2. FBE 参数] ---
# FBE：每个频带执行 带通 -> 平方 -> 滑动均值 -> 开方，输出 RMS 能量图
# 前端建议：频带数量可调整，用户可传入 BANDS 参数调整
# FBE_BANDS:
#   形式：[(low_hz, high_hz), ...]
#   要求：每段满足 0 < low < high
FBE_BANDS = [
    (1, 5),
    (5, 10),
    (10, 20),
    (20, 50),
]
FBE_RMS_WINDOW_SEC = 0.1     # RMS 窗长 (s)，越大越平滑，瞬态响应越慢
# FBE_COLORBAR_RANGE:
#   可选值：
#     - "auto": 自动按数据范围设定色标
#     - (vmin, vmax): 手动固定全局色标，便于多图横向对比
FBE_COLORBAR_RANGE = "auto"
# FBE_NYQUIST_GUARD:
#   可选值：
#     - "clip": 超 Nyquist 的高频上限自动裁剪（推荐）
#     - "skip": 超 Nyquist 的频带直接跳过
FBE_NYQUIST_GUARD = "clip"
# FBE_NYQUIST_CLIP_RATIO:
#   含义：clip 模式下实际上限 = Nyquist * ratio
#   建议：0.95~0.99，避免贴边数值不稳定
FBE_NYQUIST_CLIP_RATIO = 0.98
# FBE_STREAM_STITCH_ENABLE / FBE_STREAM_STITCH_SEC:
#   作用：流式 FBE 相邻 chunk 边界做 cross-fade，减弱竖条纹伪影
#   推荐：True，0.1~0.5 s（与 FBE_RMS_WINDOW_SEC 同量级）
FBE_STREAM_STITCH_ENABLE = True
FBE_STREAM_STITCH_SEC = 0.5

# --- [E3. Bandplots 参数] ---
# BANDS:
#   形式：[(low_hz, high_hz), ...]；用于 Figure 1 左侧 BANDPLOTS
#   前端建议：频带数量可调整，用户可传入 BANDS 参数调整
BANDS = [
    (1, 5),    # 频带 1
    (5, 10),   # 频带 2
    (10, 20),  # 频带 3
    (20, 50),  # 频带 4
]
# BANDPLOTS_ENV_BAND:
#   前端建议：用户可传入 BANDPLOTS_ENV_BAND 参数调整
BANDPLOTS_ENV_BAND = (1, 20)
# HP_REMOVE_DRIFT_HZ:
#   作用：带通前高通去漂移，抑制极低频趋势污染
#   可选：0 表示关闭；推荐范围 [0, fs/2)
HP_REMOVE_DRIFT_HZ = 1.0

# --- [E4. 滤波通用参数] ---
# DO_BAND_FILTER:
#   总滤波开关。关闭后相关滤波步骤将被跳过（用于快速对照）
DO_BAND_FILTER = True
# FILTER_ORDER:
#   候选：2/4/6/8
#   调大：截止更陡，但相位/边界效应与计算开销更高
FILTER_ORDER = 4
# STRATEGY_ZERO_PHASE:
#   True: sosfiltfilt，零相位，无时延，计算更慢
#   False: sosfilt，单向滤波，速度更快
STRATEGY_ZERO_PHASE = False

# --- [E5. 去噪与可读性增强] ---
# DO_COMMON_MODE / COMMON_MODE_METHOD:
#   作用：每个时刻按全通道做共模估计并减除（axis=0）
#   COMMON_MODE_METHOD 可选：
#     - "mean":   更快
#     - "median": 对异常值更稳健（推荐）
DO_COMMON_MODE = True
COMMON_MODE_METHOD = "median"
# DO_RAW_PER_CHANNEL_DEMEAN:
#   作用：每道减去自身时间均值（axis=1），主要提升 RAW 可读性
DO_RAW_PER_CHANNEL_DEMEAN = True

# --- [E6. Spectrogram 参数] ---
# SPECTROGRAM_TARGET_CH:
#   形式：int 或 list[int]
#   含义：指定需要绘制时频图的绝对通道号
#   前端建议：用户可传入 SPECTROGRAM_TARGET_CH 参数调整
SPECTROGRAM_TARGET_CH = [
    800, 
    900, 
    1000, 
    1100, 
    1200
    ]
# SPECTROGRAM_WINDOW_SEC:
#   可选值：None 或 正数（秒）
#   含义：谱图每一帧 STFT 的时间窗长度（以“可视化采样率 visual_fs”为基准）
#   说明：
#     - None：沿用固定点数窗（SPECTROGRAM_NPERSEG），与旧行为一致
#     - 设为秒数：按 visual_fs 自动换算成点数窗，避免不同数据长度/降采样下难以手动调参
SPECTROGRAM_WINDOW_SEC = None
# SPECTROGRAM_NPERSEG / SPECTROGRAM_OVERLAP_RATIO:
#   仅当 SPECTROGRAM_WINDOW_SEC=None 时生效
#   说明：nperseg 为每窗采样点数；overlap_ratio 为重叠比例（0~1）
SPECTROGRAM_NPERSEG = 256
SPECTROGRAM_OVERLAP_RATIO = 0.5

# --- [E7. F-K 高级可控参数] ---
# FK_PASS_DIRECTION:
#   可选值：
#     - "positive": 仅保留 k>=0（当前默认）
#     - "negative": 仅保留 k<=0
#     - "both":     保留全部 k（等价不做方向掩膜）
FK_PASS_DIRECTION = "positive"
# FK_PREFILTER_BAND_HZ:
#   可选值：None 或 (low_hz, high_hz)
#   作用：FK 前先做带通，抑制无关频段；None 表示关闭
#   注意：实际运行时会按可视化采样率自动做 Nyquist 安全钳制
FK_PREFILTER_BAND_HZ = None
# FK_NORMALIZE_MODE:
#   可选值：
#     - "none":       不归一化（默认）
#     - "zscore":     全图 z-score
#     - "percentile": 以 |P1|/|P99| 的较大值做幅值归一
FK_NORMALIZE_MODE = "none"
# FK_COLORBAR_RANGE:
#   可选值：
#     - "auto": 自动色标（默认）
#     - (vmin, vmax): 手动固定色标，便于多批次横向对比
FK_COLORBAR_RANGE = "auto"

# -----------------------------------------------------------------
# F. 显示与坐标控制
# -----------------------------------------------------------------
# X_AXIS_MODE:
#   可选值：
#     - "seconds": 相对秒
#     - "datetime": 绝对时间（推荐用于真实时序对齐）
X_AXIS_MODE = "datetime"
# Y_MODE:
#   可选值：
#     - "length": 距离轴 (m)
#     - "channel": 通道号轴
Y_MODE = "length"
# CHANNEL_LABEL_MODE:
#   仅当 Y_MODE="channel" 时生效
#   可选值：
#     - "absolute": 显示绝对通道号
#     - "relative_to_ch_start": 显示相对 CH_START 的偏移
CHANNEL_LABEL_MODE = "absolute"
# Y_TICK_INTERVAL:
#   仅在 Y_MODE="length" 时用于主刻度间隔
Y_TICK_INTERVAL = 200
# SUBPLOT_HEIGHT:
#   预留的子图高度配置（当前主要用于风格控制/后续扩展）
SUBPLOT_HEIGHT = 3.0

# -----------------------------------------------------------------
# G. 性能与采样控制（高级）
# -----------------------------------------------------------------
# ANALYSIS_DOWNSAMPLE_RATIO:
#   用途：分析链路降采样（非显示降采样），降低算法计算负荷
#   建议：1（不降）起步，再逐步提高
ANALYSIS_DOWNSAMPLE_RATIO = 1
# DOWNSAMPLE_METHOD:
#   可选值：
#     - "mean": 观察趋势更平滑（默认）
#     - "max":  保留脉冲峰值更明显
DOWNSAMPLE_METHOD = "mean"
# VIS_MAX_PIX_T / VIS_MAX_PIX_CH:
#   作用：限制静态绘图分辨率上限，避免超大图导致渲染缓慢/内存过高
VIS_MAX_PIX_T  = 2400
VIS_MAX_PIX_CH = 2000

# -----------------------------------------------------------------
# H. 本地 Matplotlib 绘图样式
# -----------------------------------------------------------------
# 说明：本地脚本生效；WebUI 通常由前端样式系统接管。
plt.rcParams.update({
    "font.size": 10, "figure.dpi": 120,
    "path.simplify": True, "path.simplify_threshold": 1.0,
    "agg.path.chunksize": 20000,
})


# =================================================================
#                          信号处理函数  
# =================================================================
def validate_manual_config():
    """校验前端手动传入参数。缺失或非法值直接报错，避免带病计算。"""
    if MANUAL_FS_HZ is None or MANUAL_FS_HZ <= 0:
        raise ValueError("MANUAL_FS_HZ must be a positive number.")
    if MANUAL_DX_M is None or MANUAL_DX_M <= 0:
        raise ValueError("MANUAL_DX_M must be a positive number.")
    if MANUAL_GAUGE_LENGTH_M is None or MANUAL_GAUGE_LENGTH_M <= 0:
        raise ValueError("MANUAL_GAUGE_LENGTH_M must be a positive number.")
    if MANUAL_N_CHANNELS_EXPECTED is None or int(MANUAL_N_CHANNELS_EXPECTED) <= 0:
        raise ValueError("MANUAL_N_CHANNELS_EXPECTED must be a positive integer.")
    if ON_CHANNEL_MISMATCH not in ("error", "skip"):
        raise ValueError('ON_CHANNEL_MISMATCH only supports "error" or "skip".')
    if CH_START < 0:
        raise ValueError("CH_START cannot be less than 0.")
    if CH_END is not None and CH_END <= CH_START:
        raise ValueError("CH_END must be greater than CH_START.")
    if CH_START >= int(MANUAL_N_CHANNELS_EXPECTED):
        raise ValueError("CH_START exceeds MANUAL_N_CHANNELS_EXPECTED.")
    if CH_END is not None and CH_END > int(MANUAL_N_CHANNELS_EXPECTED):
        raise ValueError("CH_END exceeds MANUAL_N_CHANNELS_EXPECTED.")
    if CHANNEL_LABEL_MODE not in ("absolute", "relative_to_ch_start"):
        raise ValueError('CHANNEL_LABEL_MODE only supports "absolute" or "relative_to_ch_start".')
    if LFDAS_BATCH_PADDING_MODE not in ("reflect",):
        raise ValueError('LFDAS_BATCH_PADDING_MODE only supports "reflect".')
    if LFDAS_BATCH_PAD_MAX_SEC <= 0:
        raise ValueError("LFDAS_BATCH_PAD_MAX_SEC must be positive.")
    if FK_PASS_DIRECTION not in ("positive", "negative", "both"):
        raise ValueError('FK_PASS_DIRECTION only supports "positive", "negative", or "both".')
    if FK_NORMALIZE_MODE not in ("none", "zscore", "percentile"):
        raise ValueError('FK_NORMALIZE_MODE only supports "none", "zscore", or "percentile".')
    if FK_PREFILTER_BAND_HZ is not None:
        if not isinstance(FK_PREFILTER_BAND_HZ, (list, tuple)) or len(FK_PREFILTER_BAND_HZ) != 2:
            raise ValueError("FK_PREFILTER_BAND_HZ must be None or a tuple/list with exactly two values: (low_hz, high_hz).")
        fk_low_hz, fk_high_hz = float(FK_PREFILTER_BAND_HZ[0]), float(FK_PREFILTER_BAND_HZ[1])
        if not (fk_low_hz > 0 and fk_high_hz > 0 and fk_low_hz < fk_high_hz):
            raise ValueError("FK_PREFILTER_BAND_HZ must satisfy 0 < low_hz < high_hz.")
    if FK_COLORBAR_RANGE != "auto":
        if not isinstance(FK_COLORBAR_RANGE, (list, tuple)) or len(FK_COLORBAR_RANGE) != 2:
            raise ValueError('FK_COLORBAR_RANGE must be "auto" or a tuple/list: (vmin, vmax).')
        fk_vmin, fk_vmax = float(FK_COLORBAR_RANGE[0]), float(FK_COLORBAR_RANGE[1])
        if not (np.isfinite(fk_vmin) and np.isfinite(fk_vmax) and fk_vmin < fk_vmax):
            raise ValueError("FK_COLORBAR_RANGE must satisfy finite vmin < vmax.")
    if SPECTROGRAM_WINDOW_SEC is not None and float(SPECTROGRAM_WINDOW_SEC) <= 0:
        raise ValueError("SPECTROGRAM_WINDOW_SEC must be None or a positive number (seconds).")
    if int(SPECTROGRAM_NPERSEG) < 16:
        raise ValueError("SPECTROGRAM_NPERSEG must be >= 16.")
    if not (0 < float(SPECTROGRAM_OVERLAP_RATIO) < 1):
        raise ValueError("SPECTROGRAM_OVERLAP_RATIO must satisfy 0 < value < 1.")
    if DO_BANDPLOTS:
        if not isinstance(BANDS, (list, tuple)) or len(BANDS) == 0:
            raise ValueError("BANDS must contain at least one (low_hz, high_hz) pair when DO_BANDPLOTS is enabled.")
        if not isinstance(BANDPLOTS_ENV_BAND, (list, tuple)) or len(BANDPLOTS_ENV_BAND) != 2:
            raise ValueError("BANDPLOTS_ENV_BAND must be a tuple/list with exactly two values: (low_hz, high_hz).")
        env_low_hz, env_high_hz = float(BANDPLOTS_ENV_BAND[0]), float(BANDPLOTS_ENV_BAND[1])
        if not (env_low_hz > 0 and env_high_hz > 0 and env_low_hz < env_high_hz):
            raise ValueError("BANDPLOTS_ENV_BAND must satisfy 0 < low_hz < high_hz.")


# ================== 文件读取和处理核心函数 ==================
PAT_STAMP = re.compile(r"(\d{8})_(\d{6})(?:\.(\d+))?") 
PAT_H5_MIN = re.compile(r"(\d{12})(?:_ds\d+Hz)?")

def parse_ts(timestamp_text):
    timestamp_text = timestamp_text.strip()
    if re.fullmatch(r"\d{12}", timestamp_text):
        return datetime.strptime(timestamp_text, "%Y%m%d%H%M")
    stamp_match = PAT_STAMP.fullmatch(timestamp_text)
    if stamp_match:
        return datetime.strptime(stamp_match.group(1) + stamp_match.group(2), "%Y%m%d%H%M%S")
    return None

def scan_files(input_directory_path, start_timestamp_text, end_timestamp_text):
    start_datetime, end_datetime = parse_ts(start_timestamp_text), parse_ts(end_timestamp_text)
    matched_files = []
    for file_path in sorted(Path(input_directory_path).iterdir()):
        if file_path.suffix.lower() not in (".h5", ".tdms"):
            continue
        stamp_match = PAT_STAMP.search(file_path.name)
        file_datetime = None
        if stamp_match:
            file_datetime = datetime.strptime(stamp_match.group(1) + stamp_match.group(2), "%Y%m%d%H%M%S")
        elif file_path.suffix == ".h5":
            minute_match = PAT_H5_MIN.search(file_path.name)
            if minute_match:
                file_datetime = datetime.strptime(minute_match.group(1), "%Y%m%d%H%M")
        if file_datetime and start_datetime <= file_datetime <= end_datetime:
            matched_files.append((file_datetime, file_path))
    if not matched_files:
        raise FileNotFoundError("No matching files found.")
    matched_files.sort(key=lambda item: item[0])
    return matched_files

def _h5_find_dataset(h5):
    """按优先级查找 H5 中的二维数据集，返回 (dset_path, time_first)。
    优先级: Acquisition/Raw[*]/RawData → "data" → 第一个二维 Dataset。
    与当前 compute/filter 核心模块逻辑保持对齐。"""
    if "Acquisition" in h5 and isinstance(h5["Acquisition"], h5py.Group):
        acquisition_group = h5["Acquisition"]
        raw_group_key = next(
            (group_key for group_key in acquisition_group.keys()
             if group_key.startswith("Raw") and isinstance(acquisition_group[group_key], h5py.Group)),
            None,
        )
        if raw_group_key and "RawData" in acquisition_group[raw_group_key]:
            dataset_path = f"Acquisition/{raw_group_key}/RawData"
            dataset = h5[dataset_path]
            return dataset_path, (dataset.shape[0] >= dataset.shape[1])
    if "data" in h5 and isinstance(h5["data"], h5py.Dataset) and h5["data"].ndim == 2:
        dataset = h5["data"]
        return "data", (dataset.shape[0] >= dataset.shape[1])
    first_2d_dataset_path = [None]
    def _visit(name, obj):
        if first_2d_dataset_path[0] is None and isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            first_2d_dataset_path[0] = name
    h5.visititems(_visit)
    if first_2d_dataset_path[0]:
        dataset = h5[first_2d_dataset_path[0]]
        return first_2d_dataset_path[0], (dataset.shape[0] >= dataset.shape[1])
    return None, None

def get_file_meta(file_path, expected_total_channels=None):
    """仅读取元数据（不加载数据），返回 (n_time, total_n_ch)。通道校验失败返回 None。"""
    n_time = total_n_ch = None
    if file_path.suffix.lower() == '.h5':
        with h5py.File(file_path, 'r') as h5:
            dset_path, time_first = _h5_find_dataset(h5)
            if dset_path:
                ds = h5[dset_path]
                n0, n1 = ds.shape
                if time_first:
                    n_time, total_n_ch = int(n0), int(n1)
                else:
                    n_time, total_n_ch = int(n1), int(n0)
    elif file_path.suffix.lower() == '.tdms' and HAS_TDMS:
        with TdmsFile.read(file_path) as tf:
            tdms_group = tf.groups()[0]
            channels = tdms_group.channels()
            if channels:
                total_n_ch = len(channels)
                n_time = len(channels[0])
    if n_time is None or total_n_ch is None:
        return None
    if expected_total_channels is not None and total_n_ch != int(expected_total_channels):
        return None
    return n_time, total_n_ch

def pass1_scan_streaming(file_info_list, expected_total_channels, channel_mismatch_policy):
    """流式前扫描：精确累计各文件样本数，并携带文件时间戳。
    file_info_list: [(dt, path), ...] 来自 scan_files。
    返回 file_infos: [(path, n_time, dt), ...]"""
    total_samples = 0
    file_infos = []
    skipped = []
    for file_datetime, file_path in file_info_list:
        meta = get_file_meta(file_path, expected_total_channels=expected_total_channels)
        if meta is None:
            actual_meta = get_file_meta(file_path, expected_total_channels=None)
            if actual_meta:
                _, actual_channel_count = actual_meta
                if actual_channel_count != expected_total_channels:
                    skipped.append(file_path.name)
                    if channel_mismatch_policy == "error":
                        break
            continue
        n_time, _ = meta
        total_samples += n_time
        file_infos.append((file_path, n_time, file_datetime))
    return total_samples, file_infos, skipped

def read_chunk(file_path, channel_start_index, channel_end_index, expected_total_channels=None):
    data_matrix=None
    total_n_ch = None
    n_time = None
    if file_path.suffix=='.h5':
        with open(file_path, 'rb') as file_handle:
            with h5py.File(file_handle, 'r') as h5:
                dset_path, time_first = _h5_find_dataset(h5)
                if dset_path:
                    ds = h5[dset_path]
                    n0, n1 = ds.shape
                    if time_first:
                        n_time = int(n0); total_n_ch = int(n1); data_matrix = ds[:, channel_start_index:channel_end_index].T
                    else:
                        n_time = int(n1); total_n_ch = int(n0); data_matrix = ds[channel_start_index:channel_end_index, :]
    elif file_path.suffix=='.tdms' and HAS_TDMS:
        with TdmsFile.read(file_path) as tf:
            tdms_group = tf.groups()[0]
            all_channels = tdms_group.channels()
            total_n_ch = len(all_channels)
            if total_n_ch > 0:
                n_time = len(all_channels[0])
            selected_channels = all_channels[channel_start_index:channel_end_index]
            data_matrix=np.stack([channel.data for channel in selected_channels])
    if expected_total_channels is not None and total_n_ch is not None and total_n_ch != int(expected_total_channels):
        msg = f"[Channel Check] {file_path.name} channel count mismatch: actual {total_n_ch}, expected {int(expected_total_channels)}"
        if ON_CHANNEL_MISMATCH == "error":
            raise ValueError(msg)
        print(f"[Skipped] {msg} (ON_CHANNEL_MISMATCH=skip)")
        return None, float(fs), total_n_ch, n_time
    return data_matrix, float(fs), total_n_ch, n_time

def read_window_non_stream(
    file_infos,
    window_start_sample_index,
    window_end_sample_index,
    expected_window_channel_count,
    expected_total_channels,
):
    """非流式读取：一次性拼接时间窗，返回 (ch, t) float32。"""
    target_len = window_end_sample_index - window_start_sample_index
    out = np.empty((expected_window_channel_count, target_len), dtype=np.float32)
    global_offset = 0
    write_pos = 0
    for fp, file_n_time, _ in tqdm(file_infos, desc="Non-stream concat"):
        d_cpu, _, _, _ = read_chunk(
            fp,
            CH_START,
            CH_END,
            expected_total_channels=expected_total_channels,
        )
        if d_cpu is None:
            global_offset += file_n_time
            continue
        chunk_start_idx = global_offset
        chunk_end_idx = global_offset + file_n_time
        overlap_start = max(window_start_sample_index, chunk_start_idx)
        overlap_end = min(window_end_sample_index, chunk_end_idx)
        global_offset += file_n_time
        if overlap_start >= overlap_end:
            continue
        local_s = overlap_start - chunk_start_idx
        local_e = overlap_end - chunk_start_idx
        blk = d_cpu[:, local_s:local_e].astype(np.float32, copy=False)
        take = blk.shape[1]
        out[:, write_pos:write_pos + take] = blk
        write_pos += take
        if chunk_end_idx >= window_end_sample_index:
            break
    if write_pos < target_len:
        out = out[:, :write_pos]
        print(f"[Warning] Non-stream concatenated length is shorter than expected: expected={target_len}, got={write_pos}")
    return out

class StreamAccumulator:
    def __init__(self): self.parts=[]
    def append(self, array_chunk): self.parts.append(cp.asnumpy(array_chunk) if HAS_GPU else array_chunk)
    def append_gap(self, channel_count, gap_column_count):
        if gap_column_count > 0:
            self.parts.append(np.full((channel_count, gap_column_count), np.nan, dtype=np.float32))
    def concat(self): return np.concatenate(self.parts, axis=1) if self.parts else None

# ================== 主逻辑 ==================
def main():
    validate_manual_config()
    file_info_list = scan_files(IN_DIR, START_TS, END_TS)

    expected_total_channels = int(MANUAL_N_CHANNELS_EXPECTED)
    probe_matrix = None
    probe_datetime = None
    for file_datetime, file_path in file_info_list:
        probe_chunk_matrix, probe_sample_rate_hz, _, _ = read_chunk(
            file_path, CH_START, CH_END, expected_total_channels=expected_total_channels
        )
        if probe_chunk_matrix is not None:
            probe_matrix = probe_chunk_matrix
            probe_datetime = file_datetime
            fs = probe_sample_rate_hz
            break

    if probe_matrix is None:
        print("[Error] All candidate files failed channel consistency check; cannot continue.")
        return

    sample_rate_hz = fs
    channel_count, probe_chunk_sample_count = probe_matrix.shape
    ch_end_eff = CH_END if CH_END is not None else expected_total_channels
    expected_window_n_ch = ch_end_eff - CH_START
    if channel_count != expected_window_n_ch:
        raise ValueError(
            f"空间窗口通道数异常: 实际 {channel_count}, 期望 {expected_window_n_ch} "
            f"(CH_START={CH_START}, CH_END={ch_end_eff})"
        )

    # 精确累计时间：逐文件扫描得到总样本数，避免 approx 导致时轴偏差
    exact_total_samples, file_infos, scan_skipped = pass1_scan_streaming(
        file_info_list, expected_total_channels, ON_CHANNEL_MISMATCH
    )
    
    idx_start = int(round(t_start * sample_rate_hz))
    if t_end is None:
        idx_end = exact_total_samples
    else:
        idx_end = int(round(t_end * sample_rate_hz))
    
    idx_start = max(0, idx_start)
    idx_end = min(exact_total_samples, max(idx_start, idx_end))
    target_len = idx_end - idx_start
    
    if target_len <= 0:
        print(f"[Error] Invalid time crop range: {t_start}s - {t_end}s")
        return

    start_dt_global = probe_datetime + timedelta(seconds=t_start)

    # --- 根据 TIME_AXIS_GAP_MODE 计算 r_t ---
    real_span_seconds = None
    if USE_STREAMING_PROCESSING and TIME_AXIS_GAP_MODE == "realtime" and len(file_infos) >= 2:
        first_dt = file_infos[0][2]
        last_dt  = file_infos[-1][2]
        last_dur = file_infos[-1][1] / sample_rate_hz
        real_span_seconds = (last_dt - first_dt).total_seconds() + last_dur
        real_total_vis = int(real_span_seconds * sample_rate_hz)
        r_t = max(1, int(real_total_vis / VIS_MAX_PIX_T))
    else:
        r_t = max(1, int(target_len / VIS_MAX_PIX_T))

    r_ch = max(1, int(channel_count / VIS_MAX_PIX_CH))
    
    print(f"[Config] FS={sample_rate_hz}Hz | Crop range: {t_start}s ~ {t_end if t_end else 'End'}s | Exact total samples: {exact_total_samples}")
    print(f"[Config] Downsample: T/{r_t}, CH/{r_ch} | Valid files: {len(file_infos)}")
    print(f"[Config] USE_STREAMING_PROCESSING={USE_STREAMING_PROCESSING}")
    print(f"[Config] TIME_AXIS_GAP_MODE={TIME_AXIS_GAP_MODE}"
          + (f" | real_span={real_span_seconds:.1f}s" if real_span_seconds else ""))
    print(f"[Config] DX={MANUAL_DX_M}m | GaugeLength={MANUAL_GAUGE_LENGTH_M}m | N_EXPECTED={expected_total_channels}")
    print(f"[Frontend Note] ON_CHANNEL_MISMATCH={ON_CHANNEL_MISMATCH}; in skip mode, mismatched files are recorded and skipped.")

    acc_raw = StreamAccumulator() if (DO_RAW_VIS or DO_FREQ_DIST or DO_SPECTROGRAM or DO_FK) else None
    acc_bands = [StreamAccumulator() for _ in BANDS] if DO_BANDPLOTS else []
    acc_band_env = StreamAccumulator() if DO_BANDPLOTS else None
    acc_lfdas = StreamAccumulator() if DO_LFDAS else None
    acc_fbe = [StreamAccumulator() for _ in FBE_BANDS] if DO_FBE else []
    fbe_specs = cc.build_fbe_runtime_specs(
        FBE_BANDS,
        sample_rate_hz,
        guard_policy=FBE_NYQUIST_GUARD,
        nyquist_clip_ratio=FBE_NYQUIST_CLIP_RATIO,
    ) if DO_FBE else []
    if DO_FBE:
        for s in fbe_specs:
            if s["note_text"]:
                print(f"[FBE-Guard] {s['note_text']}")
        if USE_STREAMING_PROCESSING and FBE_STREAM_STITCH_ENABLE:
            print(f"[Config] FBE Stream Stitch: {FBE_STREAM_STITCH_SEC:.2f}s")

    # --- LFDAS Overlap-Save 初始化 ---
    lfdas_pad_len = 0
    lfdas_prev_tail = None          # 上一 chunk 尾部（CPU numpy），用于下一 chunk 的左侧 padding
    if DO_LFDAS:
        if LFDAS_METHOD == "bandpass":
            _lfdas_fc = LFDAS_HIGH_HZ * LFDAS_LOW_RATIO
        else:
            _lfdas_fc = LFDAS_HIGH_HZ
        lfdas_pad_len = int(LFDAS_OVERLAP_CYCLES / max(_lfdas_fc, 1e-12) * sample_rate_hz)
        lfdas_pad_len = max(lfdas_pad_len, 1)
        print(f"[Config] LFDAS Overlap-Save: cycles={LFDAS_OVERLAP_CYCLES}, "
              f"fc={_lfdas_fc:.4g} Hz, pad={lfdas_pad_len} samples/ch "
              f"(≈{lfdas_pad_len * 4 / 1024:.1f} KB/ch)")
        if LFDAS_STREAM_STITCH_ENABLE:
            print(f"[Config] LFDAS Stream Stitch: {LFDAS_STREAM_STITCH_SEC:.2f}s")

    t0 = perf_counter()
    vis_channel_count = channel_count // r_ch
    if USE_STREAMING_PROCESSING:
        # --- 流式处理 ---
        # 只遍历 pass1_scan 验证通过的文件，避免 offset 因跳过文件而错位（断层 bug）
        global_offset = 0
        prev_file_end_dt = None
        lfdas_stitch_break = False
        fbe_stitch_break = False
        for fp, file_n_time, file_dt in tqdm(file_infos, desc="Processing"):
            # ── realtime: 在本文件数据前，插入与上一文件之间的 gap ──
            if TIME_AXIS_GAP_MODE == "realtime" and prev_file_end_dt is not None:
                gap_sec = (file_dt - prev_file_end_dt).total_seconds()
                gap_vis = max(0, int(gap_sec * sample_rate_hz / r_t))
                if gap_vis > 0:
                    print(f"  [Gap] {prev_file_end_dt:%H:%M:%S} -> {file_dt:%H:%M:%S} "
                          f"= {gap_sec:.1f}s → {gap_vis} vis cols")
                    if acc_raw:    acc_raw.append_gap(vis_channel_count, gap_vis)
                    for acc in acc_bands: acc.append_gap(vis_channel_count, gap_vis)
                    if acc_lfdas:  acc_lfdas.append_gap(vis_channel_count, gap_vis)
                    for acc in acc_fbe:   acc.append_gap(vis_channel_count, gap_vis)
                    lfdas_prev_tail = None
                    lfdas_stitch_break = True
                    fbe_stitch_break = True
            d_cpu, _, _, _ = read_chunk(
                fp, CH_START, CH_END, expected_total_channels=expected_total_channels
            )
            if d_cpu is None:
                print(f"[Warning] File passed prescan but failed at data read: {fp.name}")
                continue

            current_chunk_len = d_cpu.shape[1]
            chunk_start_idx = global_offset
            chunk_end_idx = global_offset + current_chunk_len
            overlap_start = max(idx_start, chunk_start_idx)
            overlap_end = min(idx_end, chunk_end_idx)
            global_offset += current_chunk_len

            if overlap_start >= overlap_end:
                continue

            local_start_index = overlap_start - chunk_start_idx
            local_end_index = overlap_end - chunk_start_idx
            chunk_overlap_cpu = d_cpu[:, local_start_index:local_end_index]
            chunk_overlap_gpu = cp.asarray(chunk_overlap_cpu, dtype=cp.float32) if HAS_GPU else chunk_overlap_cpu.astype(np.float32)

            if DO_COMMON_MODE:
                chunk_overlap_gpu = remove_common_mode(chunk_overlap_gpu, method=COMMON_MODE_METHOD)

            if acc_raw:
                raw_visual_chunk = (
                    chunk_overlap_gpu - chunk_overlap_gpu.mean(axis=1, keepdims=True)
                    if DO_RAW_PER_CHANNEL_DEMEAN
                    else chunk_overlap_gpu
                )
                acc_raw.append(gpu_downsample(raw_visual_chunk, r_ch, r_t))

            if DO_BANDPLOTS:
                bandpass_input_chunk = chunk_overlap_gpu
                if HP_REMOVE_DRIFT_HZ > 0:
                    bandpass_input_chunk = gpu_highpass(
                        chunk_overlap_gpu,
                        sample_rate_hz,
                        HP_REMOVE_DRIFT_HZ,
                        filter_order=2,
                    )
                for band_index, (band_low_hz, band_high_hz) in enumerate(BANDS):
                    bandpassed_chunk = gpu_bandpass(
                        bandpass_input_chunk,
                        sample_rate_hz,
                        band_low_hz,
                        band_high_hz,
                    )
                    acc_bands[band_index].append(gpu_downsample(bandpassed_chunk, r_ch, r_t))

                env_low_hz, env_high_hz = BANDPLOTS_ENV_BAND
                env_bandpassed_chunk = gpu_bandpass(
                    bandpass_input_chunk,
                    sample_rate_hz,
                    env_low_hz,
                    env_high_hz,
                )
                envelope_chunk = gpu_envelope(env_bandpassed_chunk)
                acc_band_env.append(
                    gpu_downsample(
                        get_xp(envelope_chunk).log10(envelope_chunk + 1e-9),
                        r_ch,
                        r_t,
                    )
                )

            if DO_LFDAS:
                chunk_sample_count = chunk_overlap_gpu.shape[1]
                actual_padding_count = min(lfdas_pad_len, chunk_sample_count)

                # ── 左侧 padding ──
                if lfdas_prev_tail is not None:
                    left_padding_cpu = lfdas_prev_tail[:, -actual_padding_count:]
                    left_padding_gpu = (
                        cp.asarray(left_padding_cpu, dtype=cp.float32)
                        if HAS_GPU
                        else left_padding_cpu.astype(np.float32)
                    )
                else:
                    left_padding_gpu = chunk_overlap_gpu[:, :actual_padding_count][:, ::-1]
                left_padding_columns = left_padding_gpu.shape[1]

                # ── 右侧 padding（镜像反射）──
                right_padding_gpu = chunk_overlap_gpu[:, -actual_padding_count:][:, ::-1]

                # ── 拼接 → 滤波 → 裁剪 ──
                extended_chunk_gpu = cp.concatenate([left_padding_gpu, chunk_overlap_gpu, right_padding_gpu], axis=1)

                d_lfdas_data = cc.compute_lfdas_streaming_chunk(
                    extended_chunk_matrix=extended_chunk_gpu,
                    sample_rate_hz=sample_rate_hz,
                    lfdas_method=LFDAS_METHOD,
                    lfdas_high_hz=LFDAS_HIGH_HZ,
                    lfdas_low_ratio=LFDAS_LOW_RATIO,
                    filter_order=FILTER_ORDER,
                    use_zero_phase=STRATEGY_ZERO_PHASE,
                    valid_left_padding_columns=left_padding_columns,
                    valid_chunk_samples=chunk_sample_count,
                )

                # ── 保存尾部用于下一 chunk 的左侧真实 padding（转 CPU 节省显存）──
                tail_sample_count = min(lfdas_pad_len, chunk_sample_count)
                lfdas_prev_tail = (
                    cp.asnumpy(chunk_overlap_gpu[:, -tail_sample_count:])
                    if HAS_GPU
                    else chunk_overlap_gpu[:, -tail_sample_count:].copy()
                )

                lfdas_vis = gpu_downsample(
                    d_lfdas_data,
                    r_ch,
                    r_t,
                    reduction_method="mean",
                )
                lfdas_vis_np = cp.asnumpy(lfdas_vis) if HAS_GPU else np.asarray(lfdas_vis, dtype=np.float32)
                if (
                    LFDAS_STREAM_STITCH_ENABLE
                    and not lfdas_stitch_break
                    and acc_lfdas.parts
                ):
                    blend_cols = int(round(LFDAS_STREAM_STITCH_SEC * sample_rate_hz / r_t))
                    if blend_cols >= 2:
                        cc.blend_lfdas_boundary(acc_lfdas.parts[-1], lfdas_vis_np, blend_cols)
                acc_lfdas.append(lfdas_vis_np)
                lfdas_stitch_break = False

            if DO_FBE:
                rms_list = cc.compute_fbe_streaming_chunk(
                    streaming_chunk_matrix=chunk_overlap_gpu,
                    sample_rate_hz=sample_rate_hz,
                    fbe_runtime_specs=fbe_specs,
                    smoothing_window_seconds=FBE_RMS_WINDOW_SEC,
                    filter_order=FILTER_ORDER,
                )
                for band_index, rms_matrix in enumerate(rms_list):
                    if rms_matrix is None:
                        continue
                    fbe_vis = gpu_downsample(
                        cp.asarray(rms_matrix) if HAS_GPU else rms_matrix,
                        r_ch,
                        r_t,
                        reduction_method="max",
                    )
                    fbe_vis_np = cp.asnumpy(fbe_vis) if HAS_GPU else np.asarray(fbe_vis, dtype=np.float32)
                    if (
                        FBE_STREAM_STITCH_ENABLE
                        and not fbe_stitch_break
                        and acc_fbe[band_index].parts
                    ):
                        blend_cols = int(round(FBE_STREAM_STITCH_SEC * sample_rate_hz / r_t))
                        if blend_cols >= 2:
                            cc.blend_lfdas_boundary(acc_fbe[band_index].parts[-1], fbe_vis_np, blend_cols)
                    acc_fbe[band_index].append(fbe_vis_np)
                fbe_stitch_break = False

            # 更新文件结束时间（用于下一轮 gap 计算）
            prev_file_end_dt = file_dt + timedelta(seconds=current_chunk_len / sample_rate_hz)

            del chunk_overlap_gpu
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
            if chunk_end_idx >= idx_end:
                break
    else:
        # --- 非流式处理：整窗拼接后统一处理 ---
        print("[Mode] Non-stream: call core functions from compute_core.py.")

        data_full = read_window_non_stream(
            file_infos=file_infos,
            window_start_sample_index=idx_start,
            window_end_sample_index=idx_end,
            expected_window_channel_count=expected_window_n_ch,
            expected_total_channels=expected_total_channels,
        )
        if data_full.size == 0:
            print("[Error] Non-stream concatenation produced empty data; cannot continue.")
            return

        full_window_gpu = cp.asarray(data_full, dtype=cp.float32) if HAS_GPU else data_full.astype(np.float32)
        if DO_COMMON_MODE:
            full_window_gpu = remove_common_mode(full_window_gpu, method=COMMON_MODE_METHOD)

        if acc_raw:
            raw_visual_window = (
                full_window_gpu - full_window_gpu.mean(axis=1, keepdims=True)
                if DO_RAW_PER_CHANNEL_DEMEAN
                else full_window_gpu
            )
            acc_raw.append(gpu_downsample(raw_visual_window, r_ch, r_t))

        if DO_BANDPLOTS:
            bandpass_input_window = full_window_gpu
            if HP_REMOVE_DRIFT_HZ > 0:
                bandpass_input_window = gpu_highpass(
                    full_window_gpu,
                    sample_rate_hz,
                    HP_REMOVE_DRIFT_HZ,
                    filter_order=2,
                )
            for band_index, (band_low_hz, band_high_hz) in enumerate(BANDS):
                bandpassed_window = gpu_bandpass(
                    bandpass_input_window,
                    sample_rate_hz,
                    band_low_hz,
                    band_high_hz,
                )
                acc_bands[band_index].append(gpu_downsample(bandpassed_window, r_ch, r_t))

            env_low_hz, env_high_hz = BANDPLOTS_ENV_BAND
            env_bandpassed_window = gpu_bandpass(
                bandpass_input_window,
                sample_rate_hz,
                env_low_hz,
                env_high_hz,
            )
            envelope_window = gpu_envelope(env_bandpassed_window)
            acc_band_env.append(
                gpu_downsample(get_xp(envelope_window).log10(envelope_window + 1e-9), r_ch, r_t)
            )

        full_window_cpu = cp.asnumpy(full_window_gpu) if HAS_GPU else np.asarray(full_window_gpu)

        if DO_LFDAS:
            lfdas_input_cpu = full_window_cpu
            if LFDAS_BATCH_PADDING_ENABLE:
                if LFDAS_METHOD == "bandpass":
                    lfdas_cutoff_hz = LFDAS_HIGH_HZ * LFDAS_LOW_RATIO
                else:
                    lfdas_cutoff_hz = LFDAS_HIGH_HZ
                raw_batch_pad_columns = int(
                    LFDAS_OVERLAP_CYCLES / max(lfdas_cutoff_hz, 1e-12) * sample_rate_hz
                )
                sec_capped_pad_columns = int(float(sample_rate_hz) * float(LFDAS_BATCH_PAD_MAX_SEC))
                # reflect 模式下：pad_width 必须严格小于轴长度，否则 numpy.pad 会报错
                length_capped_pad_columns = max(0, int(lfdas_input_cpu.shape[1]) - 1)
                batch_pad_columns = max(1, min(raw_batch_pad_columns, sec_capped_pad_columns, length_capped_pad_columns))
                if lfdas_input_cpu.shape[1] >= 2 and batch_pad_columns > 0:
                    lfdas_input_cpu = np.pad(
                        lfdas_input_cpu,
                        ((0, 0), (batch_pad_columns, batch_pad_columns)),
                        mode=LFDAS_BATCH_PADDING_MODE,
                    )
                    print(
                        f"[LFDAS-Batch] Applied symmetric {LFDAS_BATCH_PADDING_MODE} padding: "
                        f"left/right={batch_pad_columns} samples "
                        f"(raw={raw_batch_pad_columns}, sec_cap={sec_capped_pad_columns}, "
                        f"len_cap={length_capped_pad_columns})."
                    )
            d_lfdas = cc.compute_lfdas_batch(
                lfdas_input_cpu,
                sample_rate_hz=sample_rate_hz,
                lfdas_method=LFDAS_METHOD,
                lfdas_high_hz=LFDAS_HIGH_HZ,
                lfdas_low_ratio=LFDAS_LOW_RATIO,
                filter_order=FILTER_ORDER,
                use_zero_phase=STRATEGY_ZERO_PHASE,
            )
            if LFDAS_BATCH_PADDING_ENABLE and lfdas_input_cpu.shape[1] > full_window_cpu.shape[1]:
                batch_pad_columns = (lfdas_input_cpu.shape[1] - full_window_cpu.shape[1]) // 2
                d_lfdas = d_lfdas[:, batch_pad_columns:batch_pad_columns + full_window_cpu.shape[1]]
            acc_lfdas.append(
                gpu_downsample(
                    cp.asarray(d_lfdas) if HAS_GPU else d_lfdas,
                    r_ch,
                    r_t,
                    reduction_method="mean",
                )
            )

        if DO_FBE:
            fbe_run_bands = [(s["runtime_low_hz"], s["runtime_high_hz"]) for s in fbe_specs]
            fbe_list = cc.compute_fbe_batch(
                full_window_cpu,
                sample_rate_hz=sample_rate_hz,
                frequency_bands=fbe_run_bands,
                smoothing_window_seconds=FBE_RMS_WINDOW_SEC,
                filter_order=FILTER_ORDER,
            )
            for band_index, rms_matrix in enumerate(fbe_list):
                if not fbe_specs[band_index]["is_enabled"]:
                    continue
                if rms_matrix is None:
                    continue
                acc_fbe[band_index].append(
                    gpu_downsample(
                        cp.asarray(rms_matrix) if HAS_GPU else rms_matrix,
                        r_ch,
                        r_t,
                        reduction_method="max",
                    )
                )

        del full_window_gpu
        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()

    print(f"[Done] Elapsed: {perf_counter()-t0:.2f}s")
    if scan_skipped:
        print(f"[Frontend Note] Skipped {len(scan_skipped)} files due to channel mismatch.")
        print("[Frontend Note] skipped_files=" + ", ".join(scan_skipped))

    plot_build_config = {
        "acc_raw": acc_raw,
        "acc_bands": acc_bands,
        "acc_lfdas": acc_lfdas,
        "do_fk": DO_FK,
        "manual_dx_m": MANUAL_DX_M,
        "channel_downsample_ratio": r_ch,
        "time_downsample_ratio": r_t,
        "sample_rate_hz": sample_rate_hz,
        "ch_start": CH_START,
        "ch_end": CH_END,
        "ch_inside": CH_INSIDE,
        "do_common_mode": DO_COMMON_MODE,
        "common_mode_method": COMMON_MODE_METHOD,
        "do_raw_per_channel_demean": DO_RAW_PER_CHANNEL_DEMEAN,
        "lfdas_method": LFDAS_METHOD,
        "lfdas_high_hz": LFDAS_HIGH_HZ,
        "lfdas_low_ratio": LFDAS_LOW_RATIO,
        "filter_order": FILTER_ORDER,
        "use_zero_phase": STRATEGY_ZERO_PHASE,
        "x_axis_mode": X_AXIS_MODE,
        "y_mode": Y_MODE,
        "channel_label_mode": CHANNEL_LABEL_MODE,
        "y_tick_interval": Y_TICK_INTERVAL,
        "time_axis_gap_mode": TIME_AXIS_GAP_MODE,
        "real_span_seconds": real_span_seconds,
        "file_infos": file_infos,
        "start_dt_global": start_dt_global,
        "t_start": t_start,
        "fk_pass_direction": FK_PASS_DIRECTION,
        "fk_prefilter_band_hz": FK_PREFILTER_BAND_HZ,
        "fk_normalize_mode": FK_NORMALIZE_MODE,
        "fk_colorbar_range": FK_COLORBAR_RANGE,
        "spectrogram_window_sec": SPECTROGRAM_WINDOW_SEC,
        "spectrogram_nperseg": SPECTROGRAM_NPERSEG,
        "spectrogram_overlap_ratio": SPECTROGRAM_OVERLAP_RATIO,
    }
    plot_ctx = pl.build_plot_context(
        plot_build_config=plot_build_config,
    )
    render_config = {
        "do_bandplots": DO_BANDPLOTS,
        "do_fbe": DO_FBE,
        "do_lfdas": DO_LFDAS,
        "do_fk": DO_FK,
        "do_spectrogram": DO_SPECTROGRAM,
        "acc_bands": acc_bands,
        "bands": BANDS,
        "acc_band_env": acc_band_env,
        "bandplots_env_band": BANDPLOTS_ENV_BAND,
        "acc_fbe": acc_fbe,
        "fbe_bands": FBE_BANDS,
        "fbe_colorbar_range": FBE_COLORBAR_RANGE,
        "fbe_specs": fbe_specs,
        "fbe_rms_window_sec": FBE_RMS_WINDOW_SEC,
        "spectrogram_target_channels": SPECTROGRAM_TARGET_CH,
    }
    pl.render_enabled_figures(
        plot_context=plot_ctx,
        render_config=render_config,
    )

if __name__ == "__main__":
    main()
