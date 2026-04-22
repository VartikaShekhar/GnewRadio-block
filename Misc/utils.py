from __future__ import annotations
import matplotlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light  # 299 792 458  m/s
import datetime
from datetime import datetime, timedelta, timezone
import matplotlib.dates as mdates
from skyfield.api import wgs84, load, EarthSatellite, Topos, load_file, Angle, Distance
from skyfield.toposlib import GeographicPosition
from skyfield.timelib import Time, Timescale
import skyfield.api as api
import os
import re
import numpy as np
import pytz
from matplotlib.colors import Normalize
import csv
import pandas as pd
import pyfftw
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import kurtosis, trim_mean
from typing import Dict, List, Tuple, Set, Any, Optional
from scipy.signal.windows import blackmanharris
import matplotlib  
import time
import digital_rf as drf

matplotlib.use('Agg')  

SAT_FREQ = {"NOAA 15 [B]": 137620650, "NOAA 18 [B]": 137912500, "NOAA 19": 137100000, "STARLINK": 11.575e9, "STARLINK_DOWNCONVERTED": 1.825e9}
ground_stations = {'union_south': (+43.07154, -89.40829, 266.3), 'berkley': (37.87490231121608, -122.25777964043806, 120), 'eng': (43.07180200933477, -89.41244173090192, 267), 'mit': (42.360236, -71.089478, 98)}
# THIS_SAT = "STARLINK"
C_M_PER_S = speed_of_light
c = C_M_PER_S / 1_000  

MIN_EL = 25
BEAMWIDTH = 70  # degrees
LIST_OF_SDRS = ["07f2eed91a1c446dbae7e727112d0df7", "25d94b66e30b4397b96043246681ac14", "3e43e5a6bdee49d2ad787714161ed4c2", "2smalldb6bdee49d2ad787714161ed4c", "dt", "ground_truth"]

ts = api.load.timescale()

### ============= GROUND TRUTH PIPELINE FUNCTIONS ============= ###
def find_gt_pass(dir_of_date: str, selected_ts: str, sat_hint: str | None = None) -> str:
    """
    Pick a DRF pass directory under dir_of_date by timestamp and (optionally) satellite name.
    Example folder: STARLINK-1144_track_2025-08-09T21-25-40Z
    selected_ts format you use: '21-25-40.000000' (we'll match HH-MM-SS)
    """
    time_str = selected_ts.split(".")[0]           # '21-25-40'
    pattern = re.compile(rf".*{time_str}Z$", re.IGNORECASE)
    candidates = []
    for d in os.listdir(dir_of_date):
        full = os.path.join(dir_of_date, d)
        if not os.path.isdir(full): 
            continue
        if sat_hint and sat_hint not in d:
            continue
        if pattern.search(d):
            # must contain a DRF channel folder (e.g., Hpol)
            if any(os.path.isdir(os.path.join(full, c)) for c in ("Hpol","Vpol","ch0","ch1")):
                candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f"No DRF pass matching {time_str} (sat={sat_hint}) in {dir_of_date}")
    if len(candidates) > 1:
        # if multiple, prefer one containing the sat_hint if provided (already filtered), else first
        candidates.sort()
    return candidates[0]

def read_ground_truth(
    root: str,
    channel: str = "Hpol",
    sub: int = 0,
    offset: float = 0.0,                 # seconds from pass start
    clip_length: Optional[float] = None, # seconds to read; None = to end
    files_per_block: int = 32            # tune: 16/32/64 on a fast SSD
) -> Tuple[np.ndarray, float, int, Dict[str, Any]]:
    dio = drf.DigitalRFReader(root)
    chans = dio.get_channels()
    if not chans:
        raise RuntimeError("No DRF channels under this root.")
    if channel is None:
        for cand in ("Hpol", "Vpol", "ch0", "ch1"):
            if cand in chans: channel = cand; break
        else:
            channel = chans[0]
    elif channel not in chans:
        raise ValueError(f"Channel '{channel}' not found. Available: {chans}")

    s0, s1 = dio.get_bounds(channel)                # [inclusive, exclusive)
    props = dio.get_properties(channel)
    fs  = int(props["samples_per_second"])
    spf = int(props.get("samples_per_file", 0))     # samples per DRF file
    if spf <= 0:
        cad_ms = int(props.get("file_cadence_millisecs", 1000))
        spf = max(1, int(fs * cad_ms / 1000.0))

    start_idx = s0 + int(round(offset * fs))
    if not (s0 <= start_idx < s1):
        raise ValueError(f"Offset outside bounds ({start_idx} not in [{s0},{s1}))")
    end_idx = s1 if clip_length is None else min(s1, start_idx + int(round(clip_length * fs)))
    if end_idx <= start_idx:
        raise ValueError("Empty window after offset/clip.")
    N = end_idx - start_idx

    iq = np.empty(N, dtype=np.complex64)           # preallocate RAM
    block = spf * max(1, files_per_block)          # big, aligned reads

    t0 = time.perf_counter()
    w = 0; pos = start_idx

    # head (align to file boundary)
    head = (spf - (pos % spf)) % spf
    if head:
        n = min(head, end_idx - pos)
        iq[w:w+n] = dio.read_vector(pos, n, channel, sub)
        w += n; pos += n

    # full aligned blocks
    while pos + block <= end_idx:
        iq[w:w+block] = dio.read_vector(pos, block, channel, sub)
        w += block; pos += block

    # tail
    if pos < end_idx:
        n = end_idx - pos
        iq[w:w+n] = dio.read_vector(pos, n, channel, sub)
        w += n; pos += n

    assert w == N
    t1 = time.perf_counter()

    gib = (N * 8) / (1024**3)  # complex64 = 8 bytes/sample
    print(f"[read] {channel}:{sub} fs={fs} Hz | N={N} (~{gib:.2f} GiB) "
          f"in {t1 - t0:.1f}s → {(gib / max(t1 - t0,1e-9)):.2f} GiB/s "
          f"(spf={spf}, block={block})")

    t0_utc = datetime.fromtimestamp(start_idx / fs, tz=timezone.utc)
    t1_utc = datetime.fromtimestamp(end_idx   / fs, tz=timezone.utc)
    return iq, N / fs, fs, {"start_time_utc": t0_utc, "end_time_utc": t1_utc}


### ============= ANALYSIS FUNCTIONS ============= ###
### NOT USED
# def sharpness_score(power):
#     return np.max(power) / np.mean(power)

# def spectral_kurtosis(power):
#     return kurtosis(power, fisher=False)  # Pearson definition

# def peak_to_next_ratio(power):
#     sorted_power = np.sort(power)[::-1]
#     return sorted_power[0] / sorted_power[1] if sorted_power[1] > 0 else np.inf

# def count_peaks(power, height_ratio=0.25, distance_bins=10):
#     peak_threshold = height_ratio * np.max(power)
#     peaks, _ = find_peaks(power, height=peak_threshold, distance=distance_bins)
#     return len(peaks)

### ============= VISUALISATION ============= ###
### ============= Animation ============= ###

def animate_full_fft(iq_data: np.ndarray, 
                     fs: int, 
                     fft_size: int = 4096, 
                     overlap: int = 2048, 
                     interval_ms: int = 50, 
                     db_range: Tuple[int, int] = (-20, 20), 
                     save_path: str = None):
    print(f"animating with fft size: {fft_size}, db_range: ({db_range[0]}, {db_range[1]})")
    step = fft_size - overlap
    n_frames = (len(iq_data) - fft_size) // step

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs)) # takes sample spacing, not fs directly (cycles per sample)
    # window = np.ones(fft_size) 
    window = np.hanning(fft_size)
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot(freqs, np.full_like(freqs, db_range[0]), linewidth=0.6)

    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(*db_range)
    ax.set_xlabel("Frequency offset (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Full-Spectrum Instantaneous FFT (DC preserved)")

    def update(frame):
        start = frame * step
        seg = iq_data[start : start + fft_size] * window
        S = np.fft.fftshift(np.fft.fft(seg))
        db = 20 * np.log10(np.abs(S) + 1e-12)
        line.set_ydata(db)
        current_time = start / fs
        ax.set_title(f"Full-Spectrum Instantaneous FFT (t = {current_time:.2f} s)")
        return line,

    ani = FuncAnimation(fig, update,
                        frames=n_frames,
                        interval=interval_ms,
                        blit=False)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()

        if ext == ".mp4" and not animation.writers.is_available("ffmpeg"):
            print("[WARN] FFmpeg not available. Falling back to GIF...")
            save_path = save_path.replace(".mp4", ".gif")
            ext = ".gif"

        print(f"[INFO] Saving animation to {save_path}...")

        if ext == ".mp4":
            ani.save(save_path, writer="ffmpeg", fps=1000 // interval_ms, dpi=200)
        elif ext == ".gif":
            ani.save(save_path, writer="pillow", fps=1000 // interval_ms)
        else:
            raise ValueError(f"Unsupported animation format: {ext}")
    return ani  # Keep reference to avoid deletion
    # plt.tight_layout()
    # plt.show()


def animate_waterfall(iq_data : np.ndarray, 
                      fs : int, 
                      fft_size : int = 4096, 
                      overlap : int = 2048, 
                      n_lines : int = 300, 
                      interval_ms : int = 50, 
                      db_range : Tuple[int, int] = (-80, 20), 
                      window_type : str ='rect'):
    """
    Scrolling waterfall animation of your IQ recording.

    iq_data   : complex64 array of IQ samples
    fs        : sample rate in Hz
    fft_size  : points per FFT
    overlap   : overlap between FFTs (must be < fft_size)
    n_lines   : height of the waterfall (number of rows)
    interval_ms: update interval in milliseconds
    db_range  : (vmin, vmax) in dB for the colormap
    window_type: 'rect', 'hann', or 'blackmanharris'
    """
    step     = fft_size - overlap
    n_frames = (len(iq_data) - fft_size) // step
    if n_frames < 1:
        raise ValueError(f"Not enough data for animation ({n_frames=} frames)")

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs))

    if window_type == 'rect':
        window = np.ones(fft_size)
    elif window_type == 'hann':
        window = np.hanning(fft_size)
    elif window_type == 'blackmanharris':

        window = blackmanharris(fft_size)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    window = window / np.sqrt(np.mean(window**2))

    waterfall = np.full((n_lines, fft_size), db_range[0], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(
        waterfall,
        origin='lower',
        aspect='auto',
        extent=[freqs[0], freqs[-1], 0, n_lines],
        vmin=db_range[0],
        vmax=db_range[1],
        cmap='viridis'
    )
    ax.set_xlabel("Frequency offset (Hz)")
    ax.set_ylabel("Time slice")
    ax.set_title(f"Waterfall (window={window_type})")
    plt.colorbar(img, ax=ax, label='Power (dB)')
    
    def update(frame):
        start = frame * step
        seg   = iq_data[start : start + fft_size] * window
        S     = np.fft.fftshift(np.fft.fft(seg))
        db    = 20 * np.log10(np.abs(S) + 1e-12)

        # Scroll the waterfall buffer
        waterfall[:-1] = waterfall[1:]
        waterfall[-1]  = db

        # Compute current time in seconds
        current_time_sec = start / fs
        # ax.set_title(f"Waterfall (t = {current_time_sec:.2f} s, window={window_type})")
        title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes,
                     ha="center", va="bottom", fontsize=10)


        img.set_data(waterfall)
        return img, title_text


    ani = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval_ms,
        blit=True
    )
    plt.tight_layout()
    plt.show()

### ============= Plotting ============= ###
def plot_spectrogram(iq_data, fs, save_path,
                     fft_size=4096, overlap=3072,
                     cmap='magma', show=False, db_range=(0,20)):
    step = fft_size - overlap
    n_segments = (len(iq_data) - fft_size)//step
    window = np.hanning(fft_size)
    window /= window.sum()/fft_size   
    spec = []
    for i in range(n_segments):
        seg = iq_data[i*step : i*step+fft_size] * window
        S = np.fft.fftshift(np.fft.fft(seg))
        spec.append(20*np.log10(np.abs(S)+1e-12))
    spec = np.array(spec).T

    # auto‑scale
    pmax = spec.max()
    vmin, vmax = pmax-50, pmax

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size,1/fs))
    times = np.arange(n_segments)*(step/fs)

    plt.figure(figsize=(10,4))
    plt.imshow(spec, aspect='auto', origin='lower',
               extent=[times[0],times[-1],freqs[0],freqs[-1]],
            #    cmap=cmap, norm=Normalize(vmin=10, vmax=25))
               cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(label='Power (dB)')
    # plt.ylim(-1e4,1e4)          # zoom in around 0 Hz
    plt.xlabel("Time (s)"); plt.ylabel("Frequency offset (Hz)")
    plt.title("Compensated Spectrogram")
    plt.tight_layout()
    if show: 
        plt.show()
        plt.savefig(save_path)
        # pass
    else:   plt.savefig(save_path)

def old_plot_spectrogram(iq_data : np.ndarray, 
                    fs : int, 
                    save_path : str, 
                    fft_size : int = 4096, 
                    overlap : int = 2048,
                    cmap : str ='magma', 
                    db_range : Tuple[int, int] = (-20, 20), 
                    show : bool =False):

    print("plotting spec")
    step = fft_size - overlap
    n_segments = (len(iq_data) - fft_size) // step

    window = np.ones(fft_size)  # keep DC accurate
    spec = []

    for i in range(n_segments):
        start = i * step
        segment = iq_data[start:start + fft_size] * window
        spectrum = np.fft.fftshift(np.fft.fft(segment))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        spec.append(power_db)

    spec = np.array(spec).T  # shape: [freq, time]
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/fs))
    times = np.arange(n_segments) * (step / fs)

    plt.figure(figsize=(10, 5))
    plt.imshow(spec, aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               cmap=cmap,
               norm=Normalize(vmin=db_range[0], vmax=db_range[1])
               )
    plt.colorbar(label='Power (dB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency offset (Hz)")
    plt.title("Spectrogram (STFT View)")
    plt.tight_layout()
    if show:
        plt.show()
        plt.savefig(save_path)
        # plt.close()
    else:
        plt.savefig(save_path)
        # plt.close()
        # pass
    # # plt.show()
    # plt.savefig(o)
    # plt.close()


def plot_doppler(doppler_dict : Dict[str: List[Tuple[str,float]]], timezone : str = 'UTC', title : str = 'Doppler Shift', save_path : str = None, debug : bool = False):
  fig, ax = plt.subplots(figsize=(14, 8))

  for sat, shifts in doppler_dict.items():
      if not shifts:
          continue
      times, vals = zip(*shifts)
      times = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ')
              for t in times]
      ax.plot_date(mdates.date2num(times), vals, '-', lw=0.8, label=sat)

  # robust, no‑surprise ticks
  loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
  ax.xaxis.set_major_locator(loc)
  ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz='UTC'))

  ax.set_xlabel('Time (UTC)')
  ax.set_ylabel('Doppler Shift (Hz)')
  ax.set_title('Doppler Shifts Over Time')
  ax.grid(True)
  ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
  plt.tight_layout()
  if save_path:
    print(f"[INFO] Saving Doppler plot to {save_path}...")
    plt.savefig(save_path)
  if debug:
    plt.show()
#   plt.close()

def plot_doppler_df_or_dict(doppler_data : Dict | pd.DataFrame, timezone : str = 'UTC', title : str = 'Doppler Shift', save_path : str = None, debug : bool = False):
    """
    Plot Doppler shifts over time from either a dict or a DataFrame.

    Args:
        doppler_data: Either a dict[sat] -> list[(timestamp, shift)]
                      or a DataFrame with 'Satellite', 'Timestamp (UTC)', 'Doppler Shift (Hz)'.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    if isinstance(doppler_data, dict):
        for sat, shifts in doppler_data.items():
            if not shifts:
                continue
            times, vals = zip(*shifts)
            times = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in times]
            ax.plot_date(mdates.date2num(times), vals, '-', lw=0.8, label=sat)

    elif isinstance(doppler_data, pd.DataFrame):
        doppler_data = doppler_data.copy()
        doppler_data["Timestamp (UTC)"] = pd.to_datetime(doppler_data["Timestamp (UTC)"], utc=True)
        for sat, group in doppler_data.groupby("Satellite"):
            times = mdates.date2num(group["Timestamp (UTC)"].dt.to_pydatetime())
            shifts = group["Doppler Shift (Hz)"]
            ax.plot_date(times, shifts, '-', lw=0.8, label=sat)

    else:
        raise TypeError("doppler_data must be either a dict or a pandas DataFrame.")

    # Set up clean date formatting
    loc = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=timezone))

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Doppler Shift (Hz)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    plt.tight_layout()

    if save_path:
        print(f"[INFO] Saving Doppler plot to {save_path}...")
        plt.savefig(save_path)

    if debug:
        plt.show()

    # plt.close()


### ============= PROCESSING FILE ============= ###
def read_bin_file(exp_file_path : str, dtype : np.dtype = np.float32, sample_rate : int = 522000, clip_length : int | None = None) -> Tuple[np.ndarray, float]:
    '''
    Read binary IQ data from a file and return the data as a complex64 numpy array
    Args:
        exp_file_path: Full File Path to the binary file containing IQ data
        dtype: Data type of the binary file
        sample_rate: Sampling frequency in Hz, default is 600000.
        clip_length: Length of the clip in seconds, if None, the whole file is read
    Returns:
        data_iq: Numpy array of complex64 data
        trace_duration: Duration of the trace in seconds
    '''

    ### POTENTIAL IMPROVEMENT : USE MEMMAP FOR LARGE FILES -> DOES NOT DUMP ALL TO RAM?
    data_raw = np.fromfile(exp_file_path, dtype=dtype)

    if clip_length is not None:
        num_complex_samples = int(round(clip_length * sample_rate))
        num_float32_samples = num_complex_samples * 2  # 2 float32s = 1 complex sample
        data_raw = data_raw[:num_float32_samples]

    data_iq = data_raw.view(np.complex64)
    trace_duration = len(data_iq) / sample_rate

    if clip_length is not None:
        print(f"[INFO] Requested: {clip_length}s → Got: {trace_duration:.3f}s")

    return data_iq, trace_duration

## NOT USED, BUT MAY BE USEFUL IN THE FUTURE
def read_bin_file_remove_dc(exp_file_path, dtype=np.uint8, sample_rate=522000, clip_length=None):
    '''
    Read binary IQ data from a file and return the data as a complex64 numpy array
    Args:
        exp_file_path: Full File Path to the binary file containing IQ data
        dtype: Data type of the binary file
        sample_rate: Sampling frequency in Hz, default is 600000.
        clip_length: Length of the clip in seconds, if None, the whole file is read
    Returns:
        data_iq: Numpy array of complex64 data
        trace_duration: Duration of the trace in seconds
    '''
    data_raw = np.fromfile(exp_file_path, dtype=dtype)

    if clip_length is not None:
        num_complex_samples = int(round(clip_length * sample_rate))
        num_float32_samples = num_complex_samples * 2  # 2 float32s = 1 complex sample
        data_raw = data_raw[:num_float32_samples]

    data_iq = data_raw.view(np.complex64)
    
    trace_duration = len(data_iq) / sample_rate

    if clip_length is not None:
        print(f"[INFO] Requested: {clip_length}s → Got: {trace_duration:.3f}s")

    return data_iq, trace_duration

def get_timestamps_from_filename(filepath : str,
                                 time_delta_seconds : int,
                                 timezone_str : str ='US/Central'):
    """
    Supported filename patterns
    ---------------------------
    1. baseband_0Hz_HH-MM-SS_DD-MM-YYYY.wav          (old baseband)
    2. <uuid>_HH-MM-SS.ssssss_RX_IQ.wav              (new SDR recording)
    3. iq_<SATNAME>_DD-MM-YYYY_HH-MM-SS[.wav]        (CURRENT USED PATTERN)

    Returns
    -------
    dict with start/end times in local / UTC / unix, plus SDR & RX indices.
    """
    ts = load.timescale()   
    filename     = os.path.basename(filepath)
    parent_dir   = os.path.dirname(filepath)
    local_tz     = pytz.timezone(timezone_str)

    # -------- Pattern 1 : old baseband -----------------------------------
    m = re.search(r'baseband_(\d+)Hz_(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{4})',
                  filename)
    if m:
        time_str = m.group(2)                    # HH‑MM‑SS
        date_str = m.group(3)                    # DD‑MM‑YYYY
        dt_str   = f"{date_str} {time_str}"
        local_ts = local_tz.localize(
            datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S"))
        rx, sdr  = 0, 0

    else:
        # -------- Pattern 2 : UUID_HH-MM-SS.ssssss_RX_IQ.wav -------------
        m = re.search(r'^[a-z0-9]+_(\d{2}-\d{2}-\d{2})\.\d+_([0-9]+)_IQ',
                      filename)
        if m:
            time_str = m.group(1)                # HH‑MM‑SS
            # assume the date is encoded in the parent folder name like 17-04-2025
            date_m   = re.search(r'(\d{2}-\d{2}-\d{4})', parent_dir)
            if not date_m:
                raise ValueError("Cannot deduce recording date for SDR file.")
            date_str = date_m.group(1)
            dt_str   = f"{date_str} {time_str}"
            local_ts = local_tz.localize(
                datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S"))
            rx       = int(m.group(2))
            sdr_m    = re.search(r'SDR(\d+)', parent_dir)
            sdr      = int(sdr_m.group(1)) if sdr_m else 0

        else:
            # -------- Pattern 3 : iq_<SAT>_DD-MM-YYYY_HH-MM-SS(.wav) ------
            m = re.search(r'iq_[A-Za-z0-9\-]+_(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})',
                          filename)
            if m:
                date_str = m.group(1)            # DD‑MM‑YYYY
                time_str = m.group(2)            # HH‑MM‑SS
                dt_str   = f"{date_str} {time_str}"
                local_ts = local_tz.localize(
                    datetime.strptime(dt_str, "%d-%m-%Y %H-%M-%S"))
                rx, sdr  = 0, 0
            else:
                raise ValueError("Filename does not match any supported pattern.")

    # ---------- Convert & package results ---------------------------------
    utc_ts          = local_ts.astimezone(pytz.utc)
    unix_start      = utc_ts.timestamp()
    end_ts_utc      = utc_ts + timedelta(seconds=time_delta_seconds)
    unix_end        = end_ts_utc.timestamp()
    start_time_sf   = ts.utc(utc_ts.year, utc_ts.month, utc_ts.day,
                             utc_ts.hour, utc_ts.minute, utc_ts.second
                             + utc_ts.microsecond / 1e6)
    end_time_sf     = ts.utc(end_ts_utc.year, end_ts_utc.month, end_ts_utc.day,
                             end_ts_utc.hour, end_ts_utc.minute, end_ts_utc.second
                             + end_ts_utc.microsecond / 1e6)

    return {
        "start_time_cst":  local_ts,
        "start_time_utc":  utc_ts,
        "unix_start_time": unix_start,
        "end_time_utc":    end_ts_utc,
        "unix_end_time":   unix_end,
        "start_time_sf":   start_time_sf,
        "end_time_sf":     end_time_sf,
        "sdr":             sdr,
        "rx":              rx,
    }

def sharpness_score(power):
    return np.max(power) / np.mean(power)

def spectral_kurtosis(power):
    return kurtosis(power, fisher=False)

def peak_to_next_ratio(power):
    sorted_power = np.sort(power)[::-1]
    return sorted_power[0] / sorted_power[1] if len(sorted_power) > 1 and sorted_power[1] > 0 else np.inf


## NOT REALLY BEING USED RIGHT NOW
def analyze_fft_and_save_summary(
    freqs: np.ndarray,
    log_power_db: np.ndarray,
    satellite: str,
    timestamp_utc: str,
    duration_sec: float,
    save_dir: str,
    fft_size: int,
    Fs: int
):
    """
    Analyze an FFT spectrum and save both summary and all peaks to CSV.
    """
    # --- Summary analysis ---
    peak_idx = np.argmax(log_power_db)
    peak_freq_hz = freqs[peak_idx]
    peak_power_db = log_power_db[peak_idx]

    # Noise floor using 10% trimmed mean
    noise_floor_db = trim_mean(log_power_db, proportiontocut=0.1)
    snr_db = peak_power_db - noise_floor_db

    # Bandwidth estimate (number of bins above −3 dB from peak)
    threshold = peak_power_db - 3
    width_bins = np.sum(log_power_db >= threshold)
    bw_estimate_hz = width_bins * (Fs / fft_size)

    # Additional scores
    linear_power = 10 ** (log_power_db / 10)  # Convert dB to linear
    num_peaks = np.sum(log_power_db > noise_floor_db + 10)
    sharpness = sharpness_score(linear_power)
    kurt = spectral_kurtosis(linear_power)
    p2n = peak_to_next_ratio(linear_power)

    # Save summary row
    summary_path = os.path.join(save_dir, "fft_summary.csv")
    os.makedirs(save_dir, exist_ok=True)
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "satellite", "timestamp_utc", "fft_peak_freq_hz",
                "fft_peak_power_db", "fft_noise_floor_db",
                "snr_db", "fft_bw_estimate_hz", "pass_duration_sec",
                "num_peaks", "sharpness", "spectral_kurtosis", "peak_to_next_ratio"
            ])
        writer.writerow([
            satellite, timestamp_utc, peak_freq_hz,
            peak_power_db, noise_floor_db,
            snr_db, bw_estimate_hz, duration_sec,
            num_peaks, sharpness, kurt, p2n
        ])

    # --- Peak list ---
    peaks, _ = find_peaks(log_power_db, height=noise_floor_db, prominence=10)
    peaklist_path = os.path.join(save_dir, "fft_peak_list.csv")
    file_exists = os.path.isfile(peaklist_path)
    with open(peaklist_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "satellite", "timestamp_utc", "peak_freq_hz",
                "peak_power_db", "noise_floor_db"
            ])
        for i in peaks:
            writer.writerow([
                satellite, timestamp_utc, freqs[i],
                log_power_db[i], noise_floor_db
            ])

def merge_and_cleanup_csvs(root_dir: str,
                           wildcard: str,
                           merged_name: str = "merged.csv"):
    """
    Collect every CSV under `root_dir` that matches `wildcard`,
    concatenate them, write a single CSV → root_dir/merged_name,
    then delete the originals and prune empty folders.

    Example:
        merge_and_cleanup_csvs(OUTPUT_DIR,
                               wildcard="fft_summary.csv",
                               merged_name="ALL_fft_summaries.csv")
    """
    import os, glob, pandas as pd, shutil

    csv_files = glob.glob(os.path.join(root_dir, "**", wildcard), recursive=True)
    if not csv_files:
        print(f"[merge] No files matched {wildcard!r} — skipping.")
        return

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            # (optional) record where each row came from:
            df.insert(0, "source_file", os.path.basename(path))
            frames.append(df)
        except Exception as e:
            print(f"[merge] Could not read {path}: {e}")

    if not frames:
        print("[merge] All files failed to load — nothing to merge.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged_path = os.path.join(root_dir, merged_name)
    merged.to_csv(merged_path, index=False)
    print(f"[merge] Wrote {merged.shape[0]} rows → {merged_path}")

    for path in csv_files:
        try:
            os.remove(path)
        except OSError as e:
            print(f"[merge] Could not delete {path}: {e}")

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if dirpath == root_dir:          # never remove the root
            continue
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def find_top_satellites(
    csv_path: str,
    top_n: int = 5,
    save_path: Optional[str] = None
) -> List[str]:
    """
    Find the top N satellite entries by peak_power_db (desc) then by
    closeness of peak_freq_hz to zero (asc).

    Args:
        csv_path: Path to the CSV file containing FFT summaries.
        top_n: Number of top entries to return.
        save_path: Optional path to save the list of satellite names (one per line).

    Returns:
        A list of the top N satellite names (may include duplicates if present in the CSV).
    """
    # Load
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[find_top] No data in {csv_path}")
        return []

    # Check required columns
    for col in ("peak_power_db", "peak_freq_hz", "satellite"):
        if col not in df.columns:
            print(f"[find_top] '{col}' column not found in {csv_path}")
            return []

    # Compute absolute frequency offset
    df["abs_freq"] = df["peak_freq_hz"].abs()

    # Sort: highest power first, then smallest abs frequency
    df_sorted = df.sort_values(
        by=["peak_power_db", "abs_freq"],
        ascending=[False, True]
    )

    # Take top N
    top_df = df_sorted.head(top_n)
    top_sats = top_df["satellite"].tolist()

    # Log
    for i, row in enumerate(top_df.itertuples(), start=1):
        print(
            f"[find_top] #{i}: {row.satellite} | "
            f"power={row.peak_power_db:.2f} dB, freq={row.peak_freq_hz:.1f} Hz"
        )

    # Optional save
    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(top_sats))

    return top_sats


### ============= SIMULATION ============= ###
def check_field_of_view(ground_station : GeographicPosition , tle_file : str, start_time_sf : Time, end_time_sf : Time, min_elevation : int = 25) -> Dict[str, Dict[int, Any]]:
  '''
  Check which satellites are visible from the ground station within the specified time range.
  Args:
    ground_station: Skyfield wgs84.Topos object representing the ground station location.
    tle_file: Full Path to the TLE file, make sure it is fresh
    start_time_sf: Skyfield time object start time, UTC time 
    end_time_sf: Skyfield time object end time, UTC time.
  Returns: 
    Dictionary -> {SAT: {event: [timestamp]}}.
    timestamps are in Skyfield UTC format.
    events are: 
      0 -> rise,
      1 -> peak,
      2 -> set.
    SAT is the name of the satellite. as a string
  '''

  satellites = load.tle_file(tle_file)

  satellite_dict = {sat.name: sat for sat in satellites}
  print(len(satellite_dict))
  visible_satellites = {} # {SAT: {event: [t]}}

  print("Start Time (Skyfield UTC):", start_time_sf)
  print("End Time (Skyfield UTC):", end_time_sf)
  print(start_time_sf.utc_iso())
  print(end_time_sf.utc_iso())
  for sat_name, sat in satellite_dict.items():

    visible_satellites[sat_name] = {"obj": sat}
    relative_pos_sat = sat - ground_station
    alt0, az, dist = relative_pos_sat.at(start_time_sf).altaz()
    # print(f"SAT: {sat_name}, alt: {alt}, az: {az}, dist: {dist}")

    # events: 0->rises, 1->peaks, 2->sets
    t, events = sat.find_events(ground_station, start_time_sf, end_time_sf, altitude_degrees=min_elevation)
    if events.size != 0:
        print(f"SAT: {sat_name}, t: {t.utc_iso()}, events: {events}")
    for ti, event in zip(t, events):
      curr_val = visible_satellites[sat_name].get(event, [])
      if curr_val == []:
        visible_satellites[sat_name][event] = []

      if event == 0 or event == 1:
        visible_satellites[sat_name][event].append(ti)
      if event == 2:
        alt, az, dist = (visible_satellites[sat_name]['obj'] - ground_station).at(ti).altaz()
        if alt.degrees <= min_elevation:
          visible_satellites[sat_name][event].append(ti)

    # print(f"len of vis sats befire final: {len(list(visible_satellites.keys()))}")
    # if 1 not in visible_satellites[sat_name]:
    #     del visible_satellites[sat_name]
    if not any(k in visible_satellites[sat_name] for k in (0, 1, 2)):
        if alt0.degrees >= min_elevation:
            visible_satellites[sat_name].setdefault(0, []).append(start_time_sf)
        else:
            del visible_satellites[sat_name]
  print(f"len of vis sats after final: {len(list(visible_satellites.keys()))}")

  return visible_satellites

def doppler_calc(s: Time, e: Time, v_s: Dict[str, Dict[int, List[Time]]], 
                 observer : GeographicPosition, time_step : int = 1, f0 : float = SAT_FREQ["STARLINK"], 
                 timescale : Timescale = ts) -> Tuple[Dict[str, List[Tuple[str, float]]], 
                                                      Dict[str, List[Tuple[str, float]]],                    
                                                      Dict[str, List[Tuple[str, Angle, Angle, Distance, float]]]]: 
    
    '''
    Calculate the Doppler shift for each satellite in the visible satellites dictionary.
    Args:
        s: Start time as Time object.
        e: End time as Time object.
        v_s: Dictionary of visible satellites from check_field_of_view function, assumes that same format.
        observer: Skyfield wgs84.Topos object representing the ground station location.
        time_step: Time step in seconds for the Doppler shift calculation, minimum can be 1 second for smooth graph.
    Returns:
    '''

    doppler_shifts = {}
    all_range_rate = {}
    all_graph = {}
    # Convert input start and end times to Skyfield <Time> objects
    # start_time = ts.utc(s)
    # end_time = ts.utc(e)
    # start_time = ts.utc(datetime.utcfromtimestamp(s_unix))
    # end_time = ts.utc(datetime.utcfromtimestamp(e_unix))

    for sat, info in v_s.items():  
        starts = info.get(0, [s])  

        ends = info.get(2, [e])  # If no end times, default to end_time

        shifts = []
        these_ranges = []
        graph = []
        for t1, t2 in zip(starts, ends):  # t1 and t2 are <Time> objects # FOR IN CASE THERE ARE MULTIPLE PASSES, NOT RELAVNT FOR NOW

            current_time = t1
            while current_time.tt <= t2.tt:  # Compare using `.tt` (Terrestrial Time)
                try:
                    # Calculate Doppler shift at the current time
                    #info['obj'] == satellite object
                    pos = (info["obj"] - observer).at(current_time)
                    alt, az, dist, _, _, range_rate = pos.frame_latlon_and_rates(observer)

                    doppler_shift = f0 * ((-1 * range_rate.km_per_s) / c) 

                    # Append the current time and Doppler shift
                    shifts.append((current_time.utc_iso(), doppler_shift))
                    these_ranges.append((current_time.utc_iso(), range_rate.km_per_s))
                    graph.append((current_time.utc_iso(), alt, az, dist, doppler_shift))
                except Exception as exc:
                    print(f"Error processing satellite {sat} at {current_time.utc_iso()}: {exc}")
                    # break

                # Increment current_time by time step
                current_time = timescale.utc(current_time.utc_datetime() + timedelta(seconds=time_step))  # Add time step

            doppler_shifts[sat] = shifts
            all_range_rate[sat] = these_ranges
            all_graph[sat] = graph

    return doppler_shifts, all_range_rate, all_graph


def all_graphs_dict(all_graph: Dict[str, list[tuple[str, Angle, Angle, Distance, float]]]
                    ) -> Dict[str, Dict[str, Dict[str, object]]]:
    '''
    Convert the doppler shifts data structure to a more usable format i.e. a dictionary where each satellite has a dictionary of timestamps and their corresponding Doppler shifts.
    also subtracts the frequency f0 from each Doppler shift value.
    Args:
        doppler_shifts: Dictionary of Doppler shifts from doppler_calc function.
        f0: Frequency of the satellite in Hz, default is SAT_FREQ[THIS_SAT].
    Returns:
        {SAT: {timestamp: {"Timestamp (UTC)": ts,
                "Doppler Shift (Hz)": float(ds),
                "Elevation": alt,
                "Azimuth": az,
                "Distance (km)": dist}}}
    '''
    doppler_shifts_dict = {}
    for sat, info in all_graph.items():
        this_ds = {}
        for ts, alt, az, dist, ds in info:
            # timestamp, ds = info[0], info[1]
            this_ds[ts] =  {"Timestamp (UTC)": ts,
                "Doppler Shift (Hz)": float(ds),
                "Elevation": alt,
                "Azimuth": az,
                "Distance (km)": dist}
        doppler_shifts_dict[sat] = this_ds
    return doppler_shifts_dict

### NOT USED, BUT MAY BE USEFUL IN THE FUTURE
def save_doppler_to_csv(doppler_dict, save_path):
    """
    Accepts either of these shapes and writes a tidy CSV:

        {'SAT‑A': {'2025‑…Z': -3000.0, ...}, ...}   # dict‑of‑dicts
        {'SAT‑B': [('2025‑…Z', -3000.0), ...], ...} # dict‑of‑list‑of‑tuples
    """
    rows = []

    for sat, shifts in doppler_dict.items():
        # ─ pick the right iterator depending on the sub‑container type ─
        iterator = shifts.items() if isinstance(shifts, dict) else shifts

        for ts, shift in iterator:
            rows.append(
                {
                    "Satellite": sat,
                    "Timestamp (UTC)": ts,
                    "Doppler Shift (Hz)": float(shift),  # cast np.float64 → Python float
                }
            )

    df = pd.DataFrame(rows)



    df.to_csv(save_path, index=False)
    print(f"Saved {len(df):,} rows → {save_path}")
    return df


def save_all_graphs_to_csv(all_graphs_dict : Dict[str, Dict[str, Dict[str, object]]], save_path : str, offset : int = 0) -> pd.DataFrame:
    rows = []
    for sat, times in all_graphs_dict.items():
        for ts, entry in times.items():
            ts_dt = pd.to_datetime(ts, utc=True) + pd.Timedelta(seconds=offset)
            ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            row = {
                "Satellite": sat,
                "Timestamp (UTC)": ts_str,
                "Doppler Shift (Hz)": float(entry["Doppler Shift (Hz)"]),
                "Elevation": entry["Elevation"].degrees,
                "Azimuth": entry["Azimuth"].degrees % 360,
                "Distance (km)": entry["Distance (km)"].km,
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    df.to_csv(save_path, index=False)
    print(f"Saved {len(df):,} rows -> {save_path}")
    return df

def get_dopplers_timestamps(doppler_df : pd.DataFrame, sat : str) -> Tuple[List[datetime], List[float]]:
    """
    Convert the doppler_df DataFrame to a lists of timestamps and Doppler shifts.
    Args:
        doppler_df: DataFrame containing Doppler shifts.
    Returns:
        list of timestamps and list of Doppler shifts.
    """
    # Filter by satellite name
    # Extract and convert timestamps
    filtered_df = doppler_df[doppler_df["Satellite"] == sat]
    timestamps = filtered_df["Timestamp (UTC)"].apply(
        lambda t: pd.to_datetime(t, utc=True).to_pydatetime()
    ).tolist()

    # Extract Doppler shift values
    doppler_shifts = filtered_df["Doppler Shift (Hz)"].tolist()

    return timestamps, doppler_shifts

def run_simulation(filepath : str, 
                   tle_file_path : str, 
                   lat : float, 
                   long : float, 
                   satellite_type : str, 
                   date : str, 
                   elevation : int, 
                   timezone : str ='America/Chicago', 
                   dtype : np.dtype = np.uint8, 
                   sample_rate : int = 522000, 
                   save_path : str | None = None, 
                   clip_length : int | None = None,
                   force_start : bool = False,
                   offset : int = 0, ground_truth : bool = False) -> Tuple[
                                                            np.ndarray,                          # data_iq
                                                            float,                               # duration
                                                            pd.DataFrame,                        # all_graph_df
                                                            datetime,                            # start_time_utc
                                                            int,                                 # sdr
                                                            int,                                 # rx
                                                            Optional[Dict[str, List[Tuple[str, Angle, Angle, Distance, float]]]],  # all_graph
                                                            Optional[Dict[str, List[Tuple[str, float]]]]                           # doppler_shifts
                                                        ]:

    '''
    Main function to run the simulation.
    1. Load TLE file and check field of view for the ground station.
    2. Calculate Doppler shifts for visible satellites.
    3. Save Doppler shifts to CSV file.
    4. Plot Doppler shifts.
    '''
    ts = api.load.timescale()
    print(f"Running simulation with file: {filepath}")
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    if not ground_truth:
        data_iq, duration = read_bin_file(filepath, dtype=dtype, sample_rate=sample_rate, clip_length=clip_length)  # Load the binary file and print trace duration
        results_dict = get_timestamps_from_filename(filepath, duration, timezone) 
        sdr = results_dict['sdr']  # SDR index
        rx = results_dict['rx']  # Receiver index
        start_time_sf = results_dict['start_time_sf']  # Start time in Skyfield format
        end_time_sf = results_dict['end_time_sf']  # End time in Skyfield format
    else:
        data_iq, duration, sample_rate, results_dict = read_ground_truth(filepath, clip_length=clip_length, timezone_str=timezone)
        sdr = -1  # SDR index
        rx = -1 # Receiver index
        start_time_sf = results_dict['start_time_sf']  # Start time in Skyfield format
        end_time_sf = results_dict['end_time_sf']  # End time in Skyfield format
    # plot_spectrogram(data_iq, fs=sample_rate, save_path=os.path.join(save_path, f'0000_OG_spectrogram.png'), fft_size=2**15, overlap=512, cmap='magma', db_range=(-10, 50))
    # old_plot_spectrogram(data_iq, fs=sample_rate, save_path=os.path.join(save_path, f'0000_OG_spectrogram.png'), fft_size=2**15, overlap=512, cmap='magma', db_range=(-10, 50))

    # duration = 2*60*60  # Duration of the trace in seconds
    # duration = 2*60
    ground_station = wgs84.latlon(lat, long, elevation)
    print(f"Start time (Skyfield UTC): {start_time_sf.utc_iso()}")
    print(f"End time (Skyfield UTC): {end_time_sf.utc_iso()}")
    output_csv_path = os.path.join(save_path, 'doppler_shifts.csv')
    all_graph = None
    doppler_shifts = None
    if os.path.exists(output_csv_path) and force_start == False:
        print(f"CSV already exists. Loading from {output_csv_path}")
        all_graph_df = pd.read_csv(output_csv_path)
        print(f"Loaded {len(all_graph_df)} rows from CSV.")
        print("You can view the visualisation of the Doppler shifts in the 'doppler_plot.png' file.")
    else:
        print("CSV not found. Calculating Doppler shifts...")
        visible_satellites = check_field_of_view(ground_station, tle_file_path, start_time_sf, end_time_sf)

        if not visible_satellites:
            print("No visible satellites found.")
            return
        print(f"Visible satellites: {len(visible_satellites)}")
        # Calculate Doppler shifts
        doppler_shifts, all_range_rate, all_graph = doppler_calc(start_time_sf, end_time_sf, visible_satellites, ground_station, time_step=1)

        # Convert to dictionary format
        all_graph_dict_result = all_graphs_dict(all_graph)
        print(f"Doppler shifts calculated for {len(all_graph_dict_result)} satellites.")
        # Save to CSV
        # output_csv_path = os.path.join(save_path, f'doppler_shifts.csv')
        all_graph_df = save_all_graphs_to_csv(all_graph_dict_result, output_csv_path, offset=offset)
        plot_doppler(doppler_shifts, timezone=timezone, title=f'Doppler Shift for {satellite_type} on {date}', save_path=os.path.join(save_path, f'doppler_plot.png'))


    return data_iq, duration, all_graph_df, results_dict['start_time_utc'], sdr, rx, all_graph, doppler_shifts, sample_rate

### ============= LNB FILTERING ============= ###
@dataclass
class LNB:
    az_center: float            # degrees 0–360 (CW from North)
    beamwidth: float            # full 3-dB width, deg
    min_elev: float = 0.0       # deg above horizon
    id: str = "LNB"
    rx: str = "0"

    def _bounds(self) -> tuple[float, float]:
        half = self.beamwidth / 2
        lo   = (self.az_center - half) % 360
        hi   = (self.az_center + half) % 360
        return lo, hi

    def in_fov(self, az_deg: float, el_deg: float) -> bool:
        lo, hi = self._bounds()
        wrap   = lo > hi              # azimuth range crosses 0
        az_ok  = (lo <= az_deg <= hi) if not wrap else (az_deg >= lo or az_deg <= hi)
        return az_ok and el_deg >= self.min_elev


def filter_doppler_shifts_by_lnb(
    doppler_shifts : Dict[str, List[Tuple[str, float]]],
    all_graphs     : Dict[str, List[Tuple[str, "Angle", "Angle", "Distance", float]]],
    lnb            : LNB,
    debug: bool=False,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Return a doppler_shifts-shaped dict that keeps only those (timestamp, shift)
    pairs whose geometry (az/el) lies inside *lnb*'s field-of-view.
    """

    def _iso_s(ts: str) -> str:
        """Canonicalise ISO string → strip microsec rounding artefacts."""
        return datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc).isoformat(timespec="seconds") + "Z"

    fov_times: Dict[str, Set[str]] = {}
    for sat, samples in all_graphs.items():
        fov_times[sat] = {
            _iso_s(ts)
            for ts, alt, az, *_ in samples
            if lnb.in_fov(az.degrees % 360, alt.degrees)
        }

        if debug and sat in all_graphs:      # print first rejection, if any
            for ts, alt, az, *_ in samples:
                if _iso_s(ts) not in fov_times[sat]:
                    reason = []
                    if alt.degrees < lnb.min_elev:
                        reason.append(f"alt {alt.degrees:.1f}° < {lnb.min_elev}°")
                    lo, hi = lnb._bounds()
                    azd = az.degrees % 360
                    wrap = lo > hi
                    az_ok = (lo <= azd <= hi) if not wrap else (azd >= lo or azd <= hi)
                    if not az_ok:
                        reason.append(f"az {azd:.1f}° outside {lo:.1f}–{hi:.1f}°")
                    if reason:
                        print(f"[DEBUG] {sat}  {ts}: {'; '.join(reason)}")
                    break

    filtered: Dict[str, List[Tuple[str, float]]] = {}
    for sat, pairs in doppler_shifts.items():
        keep = [
            (ts, shift)
            for ts, shift in pairs
            if _iso_s(ts) in fov_times.get(sat, set())
        ]
        filtered[sat] = keep                    

    return filtered

def filter_doppler_df_by_lnb_df(df : pd.DataFrame, lnb : LNB, debug : bool = False):
    def in_fov(row):
        az = float(row["Azimuth"]) % 360
        el = float(row["Elevation"])
        return lnb.in_fov(az, el)

    keep_mask = df.apply(in_fov, axis=1)

    if debug:
        for _, row in df[~keep_mask].head(1).iterrows():
            az = float(row["Azimuth"]) % 360
            el = float(row["Elevation"])
            lo, hi = lnb._bounds()
            wrap = lo > hi
            az_ok = lo <= az <= hi if not wrap else az >= lo or az <= hi
            reasons = []
            if el < lnb.min_elev:
                reasons.append(f"alt {el:.1f}° < {lnb.min_elev}°")
            if not az_ok:
                reasons.append(f"az {az:.1f}° outside {lo:.1f}–{hi:.1f}°")
            print(f"[DEBUG] {row['Satellite']}  {row['Timestamp (UTC)']}: {'; '.join(reasons)}")
            break

    return df[keep_mask].reset_index(drop=True)


### ============= DATA PROCESSING ============= ###

def gpu_doppler_compensate_variable_estimate(iq_data : np.ndarray, sample_rate_hz : float, 
                                         doppler_data : np.ndarray | List, 
                                         plot_phase : bool = False, save_path : str | None = None) -> np.ndarray:
    import cupy as cp
    
    N = len(iq_data)
    t = cp.arange(N) / sample_rate_hz
    # if doppler_data already has one value per sample, just use it directly:
    if len(doppler_data) == N:
        doppler_interp = cp.asarray(doppler_data)
    else:
        # existing 1 Hz interpolation:
        doppler_time = cp.arange(len(doppler_data))      # seconds
        doppler_data_gpu = cp.asarray(doppler_data)
        doppler_interp = cp.interp(t, doppler_time, doppler_data_gpu)

    phase = 2*cp.pi * cp.cumsum(doppler_interp) / sample_rate_hz
    phase -= phase[0]

    iq_gpu = cp.asarray(iq_data)
    iq_gpu *= cp.exp(-1j * phase)
    iq_data = cp.asnumpy(iq_gpu)
    # 3) apply correction
    if save_path:
        t_cpu     = cp.asnumpy(t)
        dop_cpu   = cp.asnumpy(doppler_interp)
        phase_cpu = cp.asnumpy(phase)
        plt.figure(figsize=(12, 5))

        plt.subplot(2, 1, 1)
        plt.plot(t_cpu, dop_cpu)
        plt.title("Interpolated Doppler Shift (Hz)")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t_cpu, phase_cpu)
        plt.title("Cumulative Phase Correction (radians)")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")
        plt.grid(True)

        plt.tight_layout()
        if plot_phase:
            plt.show()
        # plt.close('all')

    return iq_data 

def doppler_compensate_variable_estimate(iq_data : np.ndarray, sample_rate_hz : float, 
                                         doppler_data : np.ndarray | List, 
                                         plot_phase : bool = False, save_path : str | None = None, gpu : bool = False) -> np.ndarray:
    if gpu:
        return gpu_doppler_compensate_variable_estimate(iq_data, sample_rate_hz, doppler_data, plot_phase, save_path)
    N = len(iq_data)
    # if doppler_data already has one value per sample, just use it directly:
    if len(doppler_data) == N:
        t = np.arange(N) / sample_rate_hz
        doppler_interp = np.asarray(doppler_data)
    else:
        # existing 1 Hz interpolation:
        t = np.arange(N) / sample_rate_hz
        doppler_time = np.arange(len(doppler_data))      # seconds
        doppler_interp = np.interp(t, doppler_time, doppler_data)
    print("len(data_iq_segment):", len(iq_data))
    print("len(doppler_theoritical):", len(doppler_data))

    # 1) interpolate Doppler to sample rate
    # doppler_interp = np.interp(
    #     t,
    #     np.linspace(0, len(doppler_data), len(doppler_data), endpoint=False),
    #     doppler_data
    # )
    # doppler_time = np.arange(len(doppler_data))  # [0,1,2,...] seconds
    # doppler_interp = np.interp(t, doppler_time, doppler_data)
    print("len(doppler inyterp):", len(doppler_interp))
    # plt.figure()
    # plt.plot(np.arange(len(doppler_data)), dop)
    # plt.title("Doppler Vector Used for Compensation")
    # plt.show()
    # 2) build & zero‐anchor phase correction
    phase = 2*np.pi * np.cumsum(doppler_interp) / sample_rate_hz
    phase -= phase[0]
    print("PHASE: ", phase[0], phase[1], phase[2])
    print("INTERP DOPPLER: ", doppler_interp[0], doppler_interp[1], doppler_interp[2])

    # 3) apply correction
    iq_data *= np.exp(-1j * phase)
    # 3) apply correction
    if save_path:
        sample_rate = sample_rate_hz
        t = np.arange(len(iq_data)) / sample_rate
        # Plot cumulative phase and its derivative (Doppler shift)
        plt.figure(figsize=(12, 5))

        plt.subplot(2, 1, 1)
        plt.plot(t, doppler_interp)
        plt.title("Interpolated Doppler Shift (Hz)")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(t, phase)
        plt.title("Cumulative Phase Correction (radians)")
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")
        plt.grid(True)

        plt.tight_layout()
        if plot_phase:
            plt.show()
        # plt.close('all')

    return iq_data  # Return modified data, though it is modified in-place

def process_data_plotting(satellite : str,
                          data_iq_corrected : np.ndarray,
                          data_iq_segment : np.ndarray,
                          avg_mag : np.ndarray,
                          avg_mag_corrected : np.ndarray,
                          avg_power_corrected : np.ndarray,
                          freqs : np.ndarray,
                          freqs2 : np.ndarray,
                          calc_fft_size : int,
                          fft_size : int,
                          sample_rate : int,
                          timestamp_theoritical : np.ndarray,
                          doppler_df : pd.DataFrame,
                          sdr : str,
                          rx : str,
                          save_path : str,
                          db_range : tuple[float, float],
                          overlap : int = 512,
                          debug : bool = False,
                          animate : bool = False
                            ):
    # if debug:
        # bin0 = calc_fft_size//2

        # # take a small neighborhood (e.g. ±10 bins) around DC for your noise floor
        # neighborhood = 10
        # noise_bins = np.concatenate([
        #     avg_power_corrected[bin0 - neighborhood : bin0],
        #     avg_power_corrected[bin0+1 : bin0 + 1 + neighborhood]
        # ])
        # noise_floor_local = np.median(noise_bins)

        # signal_power = avg_power_corrected[bin0]
        # snr_dB = 10*np.log10(signal_power / noise_floor_local)
        # peak_idx = np.argmax(avg_mag_corrected)

        # # 3) Estimate the noise floor by taking the median power
        # #    of all bins excluding, say, ±5 bins around the tone
        # exclude = 5
        # noise_bins_orig = np.concatenate((avg_mag[:peak_idx-exclude], avg_mag[peak_idx+exclude:]))
        # noise_bins_corr = np.concatenate((avg_mag_corrected[:peak_idx-exclude], avg_mag_corrected[peak_idx+exclude:]))

        # noise_floor_orig = np.median(noise_bins_orig)
        # noise_floor_corr = np.median(noise_bins_corr)

        # # 4) Compute SNR (in dB) before and after
        # snr_orig = 10 * np.log10(avg_mag[peak_idx] / noise_floor_orig)
        # snr_corr = 10 * np.log10(avg_mag_corrected[peak_idx] / noise_floor_corr)

        # print(f"Pre‑comp SNR   = {snr_orig:.2f} dB")
        # print(f"Post‑comp SNR  = {snr_corr:.2f} dB")
        # print(f"Measured gain = {snr_corr - snr_orig:.2f} dB")

    start_time = timestamp_theoritical[0]
    end_time = timestamp_theoritical[-1]
    duration_pass_sec = (end_time - start_time).total_seconds()
    duration_sec = len(data_iq_segment) / sample_rate
    duration_str = f"Duration: {duration_sec:.2f} seconds"
    if doppler_df is not None:
        filtered_df = doppler_df[doppler_df["Satellite"] == satellite]
        az = filtered_df["Azimuth"].tolist()
        el = filtered_df["Elevation"].tolist()
        az = sorted(az)
        el = sorted(el)

    # Extract Doppler shift values
    doppler_shifts = filtered_df["Doppler Shift (Hz)"].tolist()
    az_el_info_str = f"Azimuth: {az[0]:.2f} to {az[-1]:.2f} | Elevation: {el[0]:.2f} to {el[-1]:.2f} "
    duration_str = f"Pass: {start_time.strftime('%Y-%m-%d %H:%M:%S')} → {end_time.strftime('%Y-%m-%d %H:%M:%S')}  |  Duration: {duration_pass_sec:.2f} s \n {az_el_info_str}"
    plt.figure(figsize=(10, 6))
    # plt.plot(freqs, avg_mag, label="Original")
    # plt.legend()
    # plt.title(f"ORIGINAL Averaged FFT Magnitude Spectrum: {satellite} - SDR{int(sdr)} RX{int(rx)}")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Average Magnitude")

    # # Add below the plot
    # plt.figtext(0.5, 0.01, duration_str, wrap=True, horizontalalignment='center', fontsize=10)

    # # Optional: Add padding if it's getting cut off
    # plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # plt.grid(True)
    # plt.savefig(os.path.join(save_path, f'fft_plot_{satellite}.png'))
    # plt.close()
    # if debug:
    #     plt.show()
    # n_blocks = min(len(data_iq_corrected), len(data_iq_segment)) // calc_fft_size
    plt.figure(figsize=(10,4))
    # plt.rcParams.update({'font.size': 15})
    plt.plot(freqs, avg_mag, label='Raw IQ Data', alpha=0.7)
    plt.plot(freqs, avg_mag_corrected,  label='After Compensation', linewidth=2)
    plt.title(f'FFT {satellite} - (SDR{int(sdr)} RX{int(rx)})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.legend()
    plt.grid(True)
    # plt.ylim(-30, 20)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.figtext(0.5, 0.01, duration_str, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(os.path.join(save_path, f"fft_{satellite}.png"))
    plt.close("all")
    # TODO: ADD AZ AND EL INFO
    if debug:
        plt.show()
    # N_total = len(data_iq_corrected)
    # X = pyfftw.interfaces.numpy_fft.fft(data_iq_corrected, n=N_total,
    #                                 threads=5,
    #                                 planner_effort='FFTW_MEASURE')
    # power = np.abs(X)**2
    # power_db = 10*np.log10(power + 1e-12)
    # freqs = np.fft.fftshift(np.fft.fftfreq(N_total, d=1/sample_rate))
    # power_db = np.fft.fftshift(power_db)

    # # assume freqs and power_db are already defined as in your snippet
    # plt.figure(figsize=(10, 6))
    # plt.plot(freqs, power_db, linewidth=0.8)

    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power (dB)')
    # plt.title('WHOLE Doppler-Compensated Spectrum')
    # plt.grid(True)

    # # if you want to zoom in around zero Doppler, e.g. ±200 kHz:
    # # plt.xlim(-200e3, 200e3)

    # # plt.show()  
    # plt.savefig(os.path.join(save_path, f'corr_all_spec_{satellite}.png'))

    # plot_spectrogram(
    #     data_iq_corrected, sample_rate, os.path.join(save_path, f'corr_spec_plot_{satellite}.png') ,fft_size=fft_size, overlap=overlap,
    #     cmap='magma', db_range=db_range, show=debug)
    old_plot_spectrogram(data_iq_corrected, fs=sample_rate, save_path=os.path.join(save_path, f'corr_og_spec_plot_{satellite}.png'), fft_size=2**15, overlap=512, cmap='gray_r', db_range=db_range)
    
    if animate:
        animate_full_fft(data_iq_segment, fs=sample_rate, fft_size=fft_size, overlap=overlap,
                interval_ms=50,db_range=db_range, save_path=os.path.join(save_path, f'og_fft_animation_{satellite}.mp4'))
        animate_full_fft(data_iq_corrected, fs=sample_rate, fft_size=fft_size, overlap=overlap,
                        interval_ms=50,db_range=db_range, #save_path=None
                        save_path=os.path.join(save_path, f'corrected_fft_animation_{satellite}.mp4')
                        )
    # plt.close('all')  # force close previous figures
    
def full_data_fft(data: np.ndarray, Fs: float = 522000):
    import cupy as cp
    n = data.shape[0]
    eps = 1e-12

    x = cp.asarray(data, dtype=cp.complex64)
    X = cp.fft.fft(x)
    power = cp.abs(X)**2

    power = cp.asnumpy(power)          # bring back to host
    freqs = np.fft.fftfreq(n, d=1/Fs)  # fine to compute on CPU

    power_shift = np.fft.fftshift(power)
    freqs_shift = np.fft.fftshift(freqs)
    log_power_db = 10*np.log10(power_shift + eps)
    return freqs_shift, log_power_db, power_shift

def gpu_stream_fft_average(data, fft_size, batch_size, Fs):
    import cupy as cp
    print("Running incoherent fft on GPU")
    n = len(data)
    num_segments = n // fft_size

    total_power = cp.zeros(fft_size, dtype=cp.float64)
    total_segments = 0

    for start in range(0, num_segments, batch_size):
        end = min(start + batch_size, num_segments)
        batch = data[start * fft_size : end * fft_size].reshape(-1, fft_size)

        xb = cp.asarray(batch)                      
        X = cp.fft.fft(xb, axis=1)
        power = cp.abs(X) ** 2
        total_power += power.sum(axis=0, dtype=cp.float64)       
        total_segments += (end - start)

    avg_power = (total_power / max(1, total_segments)).get()    
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/Fs))
    log_power_db = 10*np.log10(np.fft.fftshift(avg_power) + 1e-12)

    mid = len(log_power_db)//2
    print(f"POWER VALUES: {log_power_db[mid-5:mid+5]}")
    return freqs, log_power_db, np.fft.fftshift(avg_power)

def stream_fft_average(data : np.ndarray | List, fft_size : int = 2**15, batch_size : int = 1000, Fs : float = 522000, gpu : bool = False):
    if fft_size == None:
        freqs, log_power_db, avg_power = full_data_fft(data, Fs=Fs)
        return freqs, log_power_db, avg_power
    if gpu:
        freqs, log_power_db, avg_power = gpu_stream_fft_average(data=data, fft_size=fft_size, batch_size=batch_size, Fs=Fs)
        return freqs, log_power_db, avg_power
    num_segments = len(data) // fft_size
    expected_gain = 10*np.log10(num_segments)
    print(f"{num_segments=}  → expected incoherent boost ≈ {expected_gain:.1f} dB")
    total_power = np.zeros(fft_size, dtype=np.float64)
    total_segments = 0

    for start in range(0, num_segments, batch_size):
        end = min(start + batch_size, num_segments)
        # Slice and reshape batch
        batch = data[start * fft_size : end * fft_size].reshape(-1, fft_size)

        # Batched FFT using numPy
        # fft_result = np.fft.fft(batch, axis=1)
        # power = np.abs(fft_result) ** 2


        # Perform batched FFT with pyFFTW multithreading
        fft_result = pyfftw.interfaces.numpy_fft.fft(batch, axis=1, threads=5, planner_effort='FFTW_MEASURE', overwrite_input=False)
        power = np.abs(fft_result) ** 2 ## TODO: remove float

        # Accumulate average
        total_power += power.sum(axis=0)
        total_segments += (end - start)
    # spectral density
    avg_power = total_power / total_segments
    epsilon = 1e-12
    log_power_db = 10 * np.log10(avg_power + epsilon)
    # freqs = np.fft.fftfreq(fft_size)
    freqs = np.fft.fftfreq(fft_size, d=1/Fs)
    freqs = np.fft.fftshift(freqs)  
    log_power_db = np.fft.fftshift(log_power_db)
    mid = len(log_power_db)//2
    print(f"POWER VALUES: {log_power_db[mid-5:mid+5]}")
    return freqs, log_power_db, avg_power

def coherent_fft(data_iq_corrected : np.ndarray, data_iq_segment : np.ndarray, fft_size : int, Fs : float, window : Optional[np.ndarray] = None):


    n_blocks = min(len(data_iq_corrected), len(data_iq_segment)) // fft_size #(len should be same anyway)
    expected_gain = 10*np.log10(n_blocks)
    print(f"{n_blocks=}  → expected coherent boost ≈ {expected_gain:.1f} dB")
    if n_blocks < 1:
        raise ValueError(f"Not enough data ({len(data_iq_segment)}) for fft_size={fft_size}")
    if window is None:
        window = np.ones(fft_size, dtype=float)
    else:
        if len(window) != fft_size:
            raise ValueError(f"window length ({len(window)}) != fft_size ({fft_size})")

    sum_uncorr = np.zeros(fft_size, dtype=complex)
    sum_corr   = np.zeros(fft_size, dtype=complex)

    for i in range(n_blocks):
        '''
        # from plot spectogram
        start = i * step
        segment = iq_data[start:start + fft_size] * window # i dont have overlap so no step here
        spectrum = np.fft.fftshift(np.fft.fft(segment))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        spec.append(power_db)
        '''
        blk_unc = data_iq_segment[i*fft_size:(i+1)*fft_size] * window
        blk_corr = data_iq_corrected[i*fft_size:(i+1)*fft_size] * window
        
        X_unc = np.fft.fft(blk_unc, n=fft_size)
        X_corr= np.fft.fft(blk_corr,n=fft_size)
        
        sum_uncorr += X_unc
        sum_corr   += X_corr

    eps = 1e-12
    psd_unc = np.abs(sum_uncorr)**2
    psd_cor = np.abs(sum_corr)**2

    log_unc = 10 * np.log10(psd_unc + eps)
    log_cor = 10 * np.log10(psd_cor + eps)

    # shift zero-freq to center
    log_unc = np.fft.fftshift(log_unc)
    log_cor = np.fft.fftshift(log_cor)
    freqs   = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/Fs))
    return freqs, log_unc, log_cor

def process_data_og(
        data_iq: np.ndarray,
        doppler_data_df: pd.DataFrame,
        recording_start_time: datetime | None = None, 
        sdr: str = '0',
        rx: str = '0',
        sample_rate: int = 522000,
        satellite: str = "STARLINK",
        save_path: str = None,
        db_range : tuple[float, float] = (-20, 60),
        fft_size : int = 2**15,
        overlap : int = 512,
        calc_fft_size : int = 1024 *10,
        animate: bool = False,
        debug : bool = False,
        gpu : bool = False,
):
    # if animate:
    #     animate_full_fft(data_iq, fs=sample_rate, fft_size=fft_size, overlap=overlap,
    #             interval_ms=50,db_range=db_range, save_path=os.path.join(save_path, f'og_fft_animation_{satellite}.mp4'))
    timestamp_theoritical, doppler_theoritical = get_dopplers_timestamps(
        doppler_data_df,
        sat=satellite
    )
    t0_sim = timestamp_theoritical[0]         
    sim_len_s = len(doppler_theoritical) 
    if False:   
        dop_clip = doppler_theoritical 
        dop_max  = np.max(dop_clip)
        dop_min  = np.min(dop_clip)
        bin_width= sample_rate / calc_fft_size
        M        = (dop_max - dop_min) / bin_width
        gain_db  = 10*np.log10(M)

        print(f"Doppler span = {dop_min:.0f} … {dop_max:.0f} Hz")
        print(f"FFT bin width= {bin_width:.1f} Hz")
        print(f"Compression ratio M = {M:.0f} bins swept")
        print(f"Expected spectral-SNR gain around {gain_db:.1f} dB")
    if recording_start_time is not None:
        offset_sec = (t0_sim - recording_start_time).total_seconds()
        if offset_sec < 0:
            raise ValueError(
                "Theoretical series starts before the IQ recording!"
            )
        offset_samples = int(round(offset_sec * sample_rate))
    else:
        offset_samples = 0

    needed_samples = int(round(sim_len_s * sample_rate))
    end_idx = offset_samples + needed_samples
    if end_idx > len(data_iq):
        end_idx = len(data_iq)

    data_iq_segment = data_iq[offset_samples:end_idx]
    start_time = timestamp_theoritical[0]
    end_time = timestamp_theoritical[-1]
    duration_pass_sec = (end_time - start_time).total_seconds()
    if duration_pass_sec < 2:
        print(f"Warning: The pass duration is too short ({duration_pass_sec:.2f} seconds).")
        print("This may lead to inaccurate Doppler compensation and FFT results, skipping processing.")
        return False

    duration_sec = len(data_iq_segment) / sample_rate    # IQ slice length
    print(f"Processing data for {satellite} from {start_time} to {end_time}, duration: {duration_pass_sec:.2f} seconds")
    data_iq_corrected = data_iq_segment.copy()
    data_iq_corrected = doppler_compensate_variable_estimate(
        data_iq_corrected, sample_rate, doppler_theoritical, 
        plot_phase=debug, save_path=os.path.join(save_path, f'phase_plot_{satellite}.png'), gpu=gpu
    )
    if debug:
        print("starting to calculate FFT")
    
    # freqs, avg_mag, avg_mag_corrected = coherent_fft(data_iq_corrected=data_iq_corrected,
    #                                                  data_iq_segment=data_iq_segment,
    #                                                  fft_size=calc_fft_size,
    #                                                  Fs=sample_rate)

    freqs, avg_mag, avg_power = stream_fft_average(data_iq_segment, fft_size=calc_fft_size, Fs=sample_rate, gpu=gpu)
    freqs2, avg_mag_corrected, avg_power_corrected = stream_fft_average(data_iq_corrected, fft_size=calc_fft_size, Fs=sample_rate, gpu=gpu)
    
    # if debug == False:
    #     analyze_fft_and_save_summary(
    #     freqs=freqs,
    #     log_power_db=avg_mag,
    #     satellite=satellite,
    #     timestamp_utc=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    #     duration_sec=duration_sec,
    #     save_dir=os.path.join(save_path, f'uncorrected_fft_summary_{satellite}.csv'),
    #     fft_size=calc_fft_size,
    #     Fs=sample_rate
    # )
    print("FFT calculated!")

    if calc_fft_size is None:
        calc_fft_size = data_iq.shape[0]  

    if debug == False:
        analyze_fft_and_save_summary(
        freqs=freqs,
        log_power_db=avg_mag_corrected,
        satellite=satellite,
        timestamp_utc=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        duration_sec=duration_sec,
        save_dir=os.path.join(save_path, f'corrected_fft_summary_{satellite}.csv'),
        fft_size=calc_fft_size,
        Fs=sample_rate
    )
    # if debug:
        # start_time = timestamp_theoritical[0]
        # end_time = timestamp_theoritical[-1]
        # duration_pass_sec = (end_time - start_time).total_seconds()
        # duration_sec = len(data_iq_segment) / sample_rate
        # duration_str = f"Duration: {duration_sec:.2f} seconds"
        # duration_str = f"Simulated Pass: {start_time.strftime('%Y-%m-%d %H:%M:%S')} → {end_time.strftime('%Y-%m-%d %H:%M:%S')}  |  Duration: {duration_pass_sec:.2f} s"
    process_data_plotting(satellite=satellite, data_iq_corrected=data_iq_corrected,
                          data_iq_segment=data_iq_segment, avg_mag=avg_mag, avg_mag_corrected=avg_mag_corrected,
                          avg_power_corrected=avg_mag_corrected, freqs=freqs, freqs2=freqs2, calc_fft_size=calc_fft_size,
                          fft_size=fft_size, sample_rate=sample_rate, timestamp_theoritical=timestamp_theoritical, doppler_df=doppler_data_df, sdr=sdr, rx=rx, 
                          save_path=save_path, db_range=db_range, debug=debug, animate=animate)
    return True


def process_data(
        data_iq: np.ndarray,
        doppler_data_df: pd.DataFrame,
        recording_start_time: datetime | None = None, 
        sdr: str = '0',
        rx: str = '0',
        sample_rate: int = 522000,
        satellite: str = "STARLINK",
        save_path: str = None,
        db_range : tuple[float, float] = (-20, 60),
        fft_size : int = 2**15,
        overlap : int = 512,
        calc_fft_size : int = 1024 *10,
        animate: bool = False,
        debug : bool = False
):
    if debug:
        plot_spectrogram(
        data_iq, sample_rate, os.path.join(save_path, f'0000_OG_spec_plot_{satellite}.png') ,fft_size=fft_size, overlap=overlap,
        cmap='magma', db_range=db_range, show=debug)
        old_plot_spectrogram(data_iq, fs=sample_rate, save_path=os.path.join(save_path, f'0000_OG_old_spec_plot_{satellite}.png'), fft_size=fft_size, overlap=overlap, cmap='gray_r', db_range=db_range)

    # if animate:
    #     animate_full_fft(data_iq, fs=sample_rate, fft_size=fft_size, overlap=overlap,
    #             interval_ms=50,db_range=db_range, save_path=os.path.join(save_path, f'og_fft_animation_{satellite}.mp4'))
    timestamp_theoritical, doppler_theoritical = get_dopplers_timestamps(
        doppler_data_df,
        sat=satellite
    )
    t0_sim = timestamp_theoritical[0]         
    sim_len_s = len(doppler_theoritical)   
    if True:   
        dop_clip = doppler_theoritical 
        dop_max = np.max(dop_clip)
        dop_min = np.min(dop_clip)
        bin_width = sample_rate / calc_fft_size 
        M = (dop_max - dop_min) / bin_width
        gain_db = 10*np.log10(M)

        print(f"Doppler span = {dop_min:.0f} … {dop_max:.0f} Hz")
        print(f"FFT bin width= {bin_width:.1f} Hz")
        print(f"Compression ratio M = {M:.0f} bins swept")
        print(f"Expected spectral-SNR gain around {gain_db:.1f} dB")
    if recording_start_time is not None:
        offset_sec = (t0_sim - recording_start_time).total_seconds()
        if offset_sec < 0:
            raise ValueError(
                "Theoretical series starts before the IQ recording!"
            )
        offset_samples = int(round(offset_sec * sample_rate))
    else:
        offset_samples = 0

    needed_samples = int(round(sim_len_s * sample_rate))
    end_idx = offset_samples + needed_samples
    if end_idx > len(data_iq):
        end_idx = len(data_iq)

    data_iq_segment = data_iq[offset_samples:end_idx]
    start_time = timestamp_theoritical[0]
    end_time = timestamp_theoritical[-1]
    duration_pass_sec = (end_time - start_time).total_seconds()
    # if duration_pass_sec < 15:
    #     print(f"Warning: The pass duration is too short ({duration_pass_sec:.2f} seconds).")
    #     print("This may lead to inaccurate Doppler compensation and FFT results.")
    #     return False

    duration_sec = len(data_iq_segment) / sample_rate    # IQ slice length
    print(f"Processing data for {satellite} from {start_time} to {end_time}, duration: {duration_pass_sec:.2f} seconds")
    data_iq_corrected = data_iq_segment.copy()
    data_iq_corrected = doppler_compensate_variable_estimate(
        data_iq_corrected, sample_rate, doppler_theoritical, 
        plot_phase=debug, save_path=os.path.join(save_path, f'phase_plot_{satellite}.png')
    )
    if debug:
        print("starting to calculate FFT")
    
    freqs, avg_mag, avg_mag_corrected = coherent_fft(data_iq_corrected=data_iq_corrected,
                                                     data_iq_segment=data_iq_segment,
                                                     fft_size=calc_fft_size,
                                                     Fs=sample_rate)

    if debug == False:
        analyze_fft_and_save_summary(
        freqs=freqs,
        log_power_db=avg_mag,
        satellite=satellite,
        timestamp_utc=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        duration_sec=duration_sec,
        save_dir=os.path.join(save_path, f'uncorrected_fft_summary_{satellite}.csv'),
        fft_size=calc_fft_size,
        Fs=sample_rate
    )
    print("FFT calculated!")


    if debug == False:
        analyze_fft_and_save_summary(
        freqs=freqs,
        log_power_db=avg_mag_corrected,
        satellite=satellite,
        timestamp_utc=start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        duration_sec=duration_sec,
        save_dir=os.path.join(save_path, f'corrected_fft_summary_{satellite}.csv'),
        fft_size=calc_fft_size,
        Fs=sample_rate
    )
    if debug:
        start_time = timestamp_theoritical[0]
        end_time = timestamp_theoritical[-1]
        duration_pass_sec = (end_time - start_time).total_seconds()
        duration_sec = len(data_iq_segment) / sample_rate
        duration_str = f"Duration: {duration_sec:.2f} seconds"
        duration_str = f"Simulated Pass: {start_time.strftime('%Y-%m-%d %H:%M:%S')} → {end_time.strftime('%Y-%m-%d %H:%M:%S')}  |  Duration: {duration_pass_sec:.2f} s"
    process_data_plotting(satellite=satellite, data_iq_corrected=data_iq_corrected,
                          data_iq_segment=data_iq_segment, avg_mag=avg_mag, avg_mag_corrected=avg_mag_corrected,
                          avg_power_corrected=avg_mag_corrected, freqs=freqs, freqs2=freqs, calc_fft_size=calc_fft_size,
                          fft_size=fft_size, sample_rate=sample_rate, timestamp_theoritical=timestamp_theoritical, sdr=sdr, rx=rx, 
                          save_path=save_path, db_range=db_range, debug=debug, animate=animate)
    return True

### ============= Helpers ============= ###
def get_tle_file_path(dir : str) -> str:
    list_of_dirs = [d for d in os.listdir(dir) if not os.path.isdir(d) and d.endswith('.tle')]
    if not list_of_dirs:
        print(f"No TLE files found in {dir}")
        return os.path.join(dir,r"tle_NOAA15_09-02-2023_04-22-36")
    return os.path.join(dir, list_of_dirs[0])  # Return the first TLE file found


def get_log_of_experiments(dir : str, dtype : np.dtype = np.uint8, sample_rate : float = 522000):
    file_list = [
        d for d in os.listdir(dir)
        if not os.path.isdir(os.path.join(dir, d))
        and not d.endswith('.csv')
        and not d.endswith('.tle')
        and not d.endswith('.png')
    ]

    records = []
    for file in file_list:
        parts = file.split('_')
        if len(parts) >= 4:
            sdr = parts[0]
            timestamp = parts[1]
            rx_port = parts[2]

            file_path = os.path.join(dir, file)

            try:
                # You provide this function
                _, duration = read_bin_file(file_path, dtype=dtype, sample_rate=sample_rate, clip_length=None)
            except Exception as e:
                print(f"Could not read {file}: {e}")
                duration = None

            records.append({
                'filename': file,
                'sdr': sdr,
                'timestamp': timestamp,
                'rx': rx_port,
                'duration_sec': duration
            })

    # Save to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(dir, 'experiment_metadata.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")

# NOT USED
def clip_iq_file(input_path, output_path, start_sec, duration_sec, sample_rate=522000, dtype=np.uint8):
    raw_data = np.fromfile(input_path, dtype=dtype)
    iq_data = raw_data.astype(np.float32).view(np.complex64)

    start_sample = int(start_sec * sample_rate)
    end_sample = int((start_sec + duration_sec) * sample_rate)

    clipped = iq_data[start_sample:end_sample]
    clipped.view(np.float32).tofile(output_path)

    print(f"Saved {len(clipped)} samples to {output_path}")
    return clipped  # Return the clipped data for further processing if needed

# NOT USED
def generate_output_filename(input_file, start_sec, rx=1, sdr_id="3e43e5a6bdee49d2ad787714161ed4c2"):
    # Parse timestamp from input filename
    base = os.path.basename(input_file)
    parent_dir = os.path.dirname(input_file)

    # Example: 3e43e5a6bdee49d2ad787714161ed4c2_11-30-50.356548_1_IQ
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError("Filename format must be: <uuid>_<HH-MM-SS.micro>_<index>_IQ")

    timestamp_str = parts[1]
    dt = datetime.strptime(timestamp_str, "%H-%M-%S.%f")

    # Add offset in seconds
    new_dt = dt + timedelta(seconds=start_sec)
    new_time_str = new_dt.strftime("%H-%M-%S.%f")

    # Construct new filename
    new_name = f"{sdr_id}_{new_time_str}_{rx}_IQ"
    new_dir = os.path.join(parent_dir, "clipped_segments")
    os.makedirs(new_dir, exist_ok=True)  # Ensure the directory exists
    return os.path.join(new_dir, new_name)

def choose_exp(selected_ts : str, selected_sdr : str, selected_rx : str, dir_of_date : str) -> Tuple[str, str, str, str, str]:

    df = pd.read_csv(os.path.join(dir_of_date, 'experiment_metadata.csv'))
    # 0.521
    mask = (
        (df['timestamp'] == selected_ts) &
        (df['sdr'].str.strip().eq(selected_sdr)) &
        (df['rx'] == selected_rx)          # make sure dtypes match!
    )

    matches = df.loc[mask]

    if matches.empty:
        raise ValueError(
            "No matching rows – double-check dtypes and values:\n"
            f"  SELECTED_TS  = {selected_ts!r}\n"
            f"  SELECTED_SDR = {selected_sdr!r}\n"
            f"  SELECTED_RX  = {selected_rx!r}\n"
            f"  First few rows:\n{df.head()}"
        )

    selected_row = matches.iloc[0]
    # DATA_FILE    = selected_row['filename']
    # print("Selected data file:", DATA_FILE)
    # selected_row = df[
    #     (df['timestamp'] == SELECTED_TS) &
    #     (df['sdr'] == SELECTED_SDR) &
    #     (df['rx'] == SELECTED_RX)
    # ].iloc[0]
    DATA_FILE = selected_row['filename']
    print(f"Selected data file: {DATA_FILE}")
    DATA_FILE_PATH = os.path.join(dir_of_date, DATA_FILE)
    # DATA_FILE_PATH = os.path.join(DIR_OF_DATE, r"iq_NOAA15_02-09-2023_06-04-49")


    print(f"Data file path: {DATA_FILE_PATH}")
    FULL_OUTPUT_DIR = os.path.join(dir_of_date, f'output_{selected_sdr}_{selected_rx}')  # Full path to the output directory
    os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)  # Create the output directory if it doesn't exist
    return DATA_FILE_PATH, FULL_OUTPUT_DIR, selected_row['duration_sec'], selected_row['sdr'], selected_row['rx']




LNB_LOOKUP: dict[tuple[str, str], LNB] = {
    ("3e43e5a6bdee49d2ad787714161ed4c2", "1"): LNB(az_center=0, beamwidth=BEAMWIDTH,  min_elev=MIN_EL), # 4 (North)
    ("07f2eed91a1c446dbae7e727112d0df7", "1"): LNB(az_center=35, beamwidth=BEAMWIDTH, min_elev=MIN_EL), # 5 ??
    ("07f2eed91a1c446dbae7e727112d0df7", "0"): LNB(az_center=70, beamwidth=BEAMWIDTH, min_elev=MIN_EL), # 6 ??
    ("25d94b66e30b4397b96043246681ac14", "0"): LNB(az_center=90, beamwidth=BEAMWIDTH, min_elev=MIN_EL), # 1 (up)
    ("25d94b66e30b4397b96043246681ac14", "1"): LNB(az_center=105, beamwidth=BEAMWIDTH, min_elev=MIN_EL), # 2
    ("3e43e5a6bdee49d2ad787714161ed4c2", "0"): LNB(az_center= 140, beamwidth=BEAMWIDTH,  min_elev=MIN_EL), # 3 
} 
