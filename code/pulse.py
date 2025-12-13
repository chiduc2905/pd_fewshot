"""
Enhanced Partial Discharge Pulse Detection and Extraction
=========================================================

This module provides advanced pulse detection and extraction for Partial Discharge (PD) signals
with precise timing control for research and analysis applications.

Key Features:
- Automatic peak detection with configurable thresholds
- Fixed-duration pulse extraction (1024 samples = 6.4µs)
- Peak positioning at specified time offset (1.25µs)
- High-quality visualization with IEEE standards
- Batch processing for multiple PD files

Author: PD Analysis Team
Date: 2024
Application: Electrical Insulation Condition Monitoring
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ADAPTIVE CONFIGURATION FOR DIFFERENT PD TYPES
# ============================================================================
PD_TYPES_CONFIG = {
    'corona': {
        'description': 'Corona Discharge - High density, small amplitude pulses',
        'prominence_factor': 0.05,      # Nhạy hơn để bắt xung nhỏ
        'height_factor': 0.1,           # Ngưỡng thấp hơn (10% max)
        'distance_us': 0.5,             # Khoảng cách tối thiểu 0.5 µs (rất ngắn)
        'width_range': (1, 50),         # Xung rất hẹp (samples)
        'detect_negative': False,       # Corona chủ yếu dương
        'adaptive_threshold': True,
        'local_std_window': 100         # Dùng std cục bộ
    },
    'surface': {
        'description': 'Surface Discharge - Irregular, clustered pulses',
        'prominence_factor': 0.15,
        'height_factor': 0.15,
        'distance_us': 1.0,             # Khoảng cách 1 µs
        'width_range': (2, 100),
        'detect_negative': True,        # Có thể có cả xung âm
        'adaptive_threshold': True,
        'local_std_window': 200
    },
    'void': {
        'description': 'Void Discharge - Symmetric positive/negative pulses',
        'prominence_factor': 0.2,
        'height_factor': 0.15,
        'distance_us': 2.0,             # Khoảng cách lớn hơn
        'width_range': (5, 150),
        'detect_negative': True,        # BẮT BUỘC phát hiện xung âm
        'adaptive_threshold': True,
        'local_std_window': 300,
        'symmetry_check': True          # Kiểm tra tính đối xứng
    },
    'default': {
        'description': 'Default configuration (backward compatible)',
        'prominence_factor': 0.1,
        'height_factor': 0.2,
        'distance_us': 1.0,
        'width_range': (2, 100),
        'detect_negative': False,
        'adaptive_threshold': False,
        'local_std_window': 100
    }
}


def calculate_local_threshold(signal_data, window_size=100):
    """
    Tính ngưỡng adaptive dựa trên std cục bộ.

    Lý thuyết: Nhiễu nền thay đổi theo vùng trong tín hiệu PD,
    nên cần ngưỡng adaptive thay vì dùng std toàn tín hiệu.

    Parameters:
    ----------
    signal_data : array-like
        Input signal
    window_size : int
        Size of the local window

    Returns:
    -------
    array-like
        Local standard deviation for each point
    """
    local_std = np.zeros_like(signal_data)
    half_window = window_size // 2

    for i in range(len(signal_data)):
        start = max(0, i - half_window)
        end = min(len(signal_data), i + half_window)
        local_std[i] = np.std(signal_data[start:end])

    return local_std


def detect_pd_peaks(signal_data, time_data, prominence_factor=None, distance_factor=None,
                    height_factor=None, filter_signal=True, pd_type='default'):
    """
    Detect peaks in Partial Discharge signals with ADAPTIVE configuration.

    Parameters:
    ----------
    signal_data : array-like
        Input PD signal amplitude data
    time_data : array-like
        Corresponding time vector in seconds
    prominence_factor : float, optional
        Peak prominence threshold as factor of signal std (overrides pd_type config)
    distance_factor : float, optional
        NOT USED - kept for backward compatibility
    height_factor : float, optional
        Minimum peak height as factor of signal max (overrides pd_type config)
    filter_signal : bool
        Apply Savitzky-Golay filter before peak detection (default: True)
    pd_type : str
        Type of partial discharge: 'corona', 'surface', 'void', or 'default'

    Returns:
    -------
    dict
        {
            'positive_peaks': (indices, times, amplitudes),
            'negative_peaks': (indices, times, amplitudes) or None,
            'all_peaks': (indices, times, amplitudes),
            'filtered_signal': filtered signal for visualization,
            'config_used': configuration dictionary
        }

    Notes:
    -----
    NEW ADAPTIVE FEATURES:
    - Uses PD-type specific configuration (corona/surface/void)
    - Min distance based on REAL TIME (microseconds) not samples
    - Detects NEGATIVE peaks for void discharge
    - Local adaptive threshold for better noise handling
    """
    # Get PD type configuration
    if pd_type not in PD_TYPES_CONFIG:
        print(f"⚠️  Warning: Unknown PD type '{pd_type}', using 'default'")
        pd_type = 'default'

    config = PD_TYPES_CONFIG[pd_type].copy()

    # Allow manual override
    if prominence_factor is not None:
        config['prominence_factor'] = prominence_factor
    if height_factor is not None:
        config['height_factor'] = height_factor

    # Calculate sampling parameters
    dt = float(np.mean(np.diff(time_data)))
    fs = 1 / dt

    print(f"\n{'='*70}")
    print(f"ADAPTIVE PEAK DETECTION - {pd_type.upper()} DISCHARGE")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"\nSignal parameters:")
    print(f"├── Length: {len(signal_data)} samples")
    print(f"├── Sampling frequency: {fs/1e6:.2f} MHz")
    print(f"└── Duration: {(time_data[-1] - time_data[0])*1e6:.1f} µs")

    # Optional preprocessing with Savitzky-Golay filter for visualization only
    if filter_signal and len(signal_data) > 21:
        window_length = min(21, len(signal_data) // 10)
        if window_length % 2 == 0:
            window_length += 1
        filtered_signal = savgol_filter(signal_data, window_length, 3)
        print(f"\n✓ Applied Savitzky-Golay filter (window: {window_length}) for visualization")
    else:
        filtered_signal = signal_data.copy()

    # ===== CALCULATE THRESHOLDS =====
    signal_max = np.max(np.abs(signal_data))

    if config['adaptive_threshold']:
        local_std = calculate_local_threshold(signal_data, config['local_std_window'])
        prominence_threshold = config['prominence_factor'] * np.median(local_std)
        print(f"✓ Using ADAPTIVE threshold (local std with window={config['local_std_window']})")
        print(f"  Median local std: {np.median(local_std):.6f} V")
    else:
        signal_std = np.std(signal_data)
        prominence_threshold = config['prominence_factor'] * signal_std
        print(f"✓ Using GLOBAL threshold")
        print(f"  Signal std: {signal_std:.6f} V")

    height_threshold = config['height_factor'] * signal_max

    # ★ CRITICAL: Min distance based on REAL TIME (microseconds), not samples!
    min_distance_samples = int(config['distance_us'] * 1e-6 * fs)

    print(f"\nDetection configuration:")
    print(f"├── Prominence factor: {config['prominence_factor']}")
    print(f"├── Height factor: {config['height_factor']}")
    print(f"├── Min distance: {config['distance_us']} µs = {min_distance_samples} samples")
    print(f"├── Detect negative: {config['detect_negative']}")
    print(f"└── Width range: {config['width_range']} samples")

    print(f"\nThresholds:")
    print(f"├── Prominence: {prominence_threshold:.6f} V")
    print(f"└── Height: {height_threshold:.6f} V ({config['height_factor']*100:.0f}% of max)")

    # ===== DETECT POSITIVE PEAKS =====
    print(f"\n{'─'*70}")
    print("DETECTING POSITIVE PEAKS...")
    print(f"{'─'*70}")

    positive_peaks, positive_props = find_peaks(
        signal_data,
        prominence=prominence_threshold,
        distance=min_distance_samples,
        height=height_threshold,
        width=config['width_range']
    )

    positive_times = time_data[positive_peaks]
    positive_amplitudes = signal_data[positive_peaks]

    print(f"✓ Found {len(positive_peaks)} positive peaks")
    if len(positive_peaks) > 0:
        print(f"  ├── Amplitude range: {np.min(positive_amplitudes):.6f} to {np.max(positive_amplitudes):.6f} V")
        print(f"  ├── Time range: {np.min(positive_times)*1e6:.1f} to {np.max(positive_times)*1e6:.1f} µs")
        if len(positive_peaks) > 1:
            print(f"  └── Avg spacing: {np.mean(np.diff(positive_times))*1e6:.2f} µs")

    # ===== DETECT NEGATIVE PEAKS (if enabled) =====
    negative_peaks = np.array([])
    negative_times = np.array([])
    negative_amplitudes = np.array([])

    if config['detect_negative']:
        print(f"\n{'─'*70}")
        print("DETECTING NEGATIVE PEAKS...")
        print(f"{'─'*70}")

        # Invert signal to detect negative peaks
        inverted_signal = -signal_data

        negative_peaks, negative_props = find_peaks(
            inverted_signal,
            prominence=prominence_threshold,
            distance=min_distance_samples,
            height=height_threshold,
            width=config['width_range']
        )

        negative_times = time_data[negative_peaks]
        negative_amplitudes = signal_data[negative_peaks]  # Use original (negative values)

        print(f"✓ Found {len(negative_peaks)} negative peaks")
        if len(negative_peaks) > 0:
            print(f"  ├── Amplitude range: {np.max(negative_amplitudes):.6f} to {np.min(negative_amplitudes):.6f} V")
            print(f"  ├── Time range: {np.min(negative_times)*1e6:.1f} to {np.max(negative_times)*1e6:.1f} µs")
            if len(negative_peaks) > 1:
                print(f"  └── Avg spacing: {np.mean(np.diff(negative_times))*1e6:.2f} µs")

    # ===== COMBINE ALL PEAKS =====
    all_indices = positive_peaks
    all_times = positive_times
    all_amplitudes = positive_amplitudes

    if len(negative_peaks) > 0:
        all_indices = np.concatenate([positive_peaks, negative_peaks])
        all_times = np.concatenate([positive_times, negative_times])
        all_amplitudes = np.concatenate([positive_amplitudes, negative_amplitudes])

        # Sort by time
        sort_idx = np.argsort(all_times)
        all_indices = all_indices[sort_idx]
        all_times = all_times[sort_idx]
        all_amplitudes = all_amplitudes[sort_idx]

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print(f"DETECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Positive peaks: {len(positive_peaks)}")
    print(f"Negative peaks: {len(negative_peaks)}")
    print(f"TOTAL PEAKS: {len(all_indices)}")
    print(f"{'='*70}\n")

    # Return dict for backward compatibility and new features
    result = {
        'positive_peaks': (positive_peaks, positive_times, positive_amplitudes),
        'negative_peaks': (negative_peaks, negative_times, negative_amplitudes) if len(negative_peaks) > 0 else None,
        'all_peaks': (all_indices, all_times, all_amplitudes),
        'filtered_signal': filtered_signal,
        'config_used': config
    }

    # For backward compatibility, also return as tuple
    return all_indices, all_times, all_amplitudes, filtered_signal


def extract_pulse_around_peak(signal_data, time_data, peak_index,
                             pulse_duration_us=6.4, peak_position_us=1.25):
    """
    Extract fixed-duration pulse centered around detected peak.

    Parameters:
    ----------
    signal_data : array-like
        Input signal data
    time_data : array-like
        Corresponding time vector
    peak_index : int
        Index of the peak in the signal
    pulse_duration_us : float
        Total pulse duration in microseconds (default: 6.4µs)
    peak_position_us : float
        Position of peak within extracted pulse in microseconds (default: 1.25µs)

    Returns:
    -------
    tuple
        (pulse_signal, pulse_time, extraction_info)

    Notes:
    -----
    The function extracts exactly 1024 samples representing 6.4µs duration
    with the peak positioned at 1.25µs from the start of the pulse.
    Time axis is NORMALIZED to start at 0 µs.
    """
    # Calculate sampling parameters
    dt = float(np.mean(np.diff(time_data)))
    fs = 1 / dt

    # Calculate required samples
    total_samples = 1024  # Fixed at 1024 samples for 6.4µs at 160MHz
    pre_peak_samples = int(peak_position_us * 1e-6 * fs)  # Should be 200 samples for 1.25µs
    post_peak_samples = total_samples - pre_peak_samples  # Should be 824 samples

    # Calculate extraction boundaries
    start_index = peak_index - pre_peak_samples
    end_index = start_index + total_samples

    # Handle boundary conditions
    signal_length = len(signal_data)
    actual_peak_position_in_pulse = pre_peak_samples  # This will be updated if padding is needed

    if start_index < 0:
        # Peak too close to beginning - pad with zeros at the start
        padding_start = abs(start_index)
        start_index = 0
        pulse_signal = np.zeros(total_samples)
        available_samples = min(total_samples - padding_start, signal_length)
        pulse_signal[padding_start:padding_start + available_samples] = signal_data[0:available_samples]

        # ★ CRITICAL FIX: Update actual peak position after padding
        actual_peak_position_in_pulse = pre_peak_samples + padding_start

        extraction_info = {
            'boundary_condition': 'start_padding',
            'padding_samples': padding_start,
            'peak_position_in_pulse': actual_peak_position_in_pulse
        }

    elif end_index >= signal_length:
        # Peak too close to end - pad with zeros at the end
        available_samples = signal_length - start_index
        pulse_signal = np.zeros(total_samples)
        pulse_signal[0:available_samples] = signal_data[start_index:signal_length]

        # Peak position remains the same as pre_peak_samples (no shift at start)
        actual_peak_position_in_pulse = pre_peak_samples

        extraction_info = {
            'boundary_condition': 'end_padding',
            'padding_samples': total_samples - available_samples,
            'peak_position_in_pulse': actual_peak_position_in_pulse
        }

    else:
        # Normal extraction - peak well within signal bounds
        pulse_signal = signal_data[start_index:end_index]
        actual_peak_position_in_pulse = pre_peak_samples

        extraction_info = {
            'boundary_condition': 'normal',
            'padding_samples': 0,
            'peak_position_in_pulse': actual_peak_position_in_pulse
        }

    # Ensure exactly 1024 samples
    if len(pulse_signal) != 1024:
        if len(pulse_signal) > 1024:
            pulse_signal = pulse_signal[:1024]
        else:
            # Pad with zeros if needed
            padding_needed = 1024 - len(pulse_signal)
            pulse_signal = np.pad(pulse_signal, (0, padding_needed), mode='constant')

    # ★ CREATE NORMALIZED TIME VECTOR: 0 to 6.4 µs
    pulse_time = np.linspace(0, pulse_duration_us * 1e-6, total_samples)

    # Add extraction statistics with ACTUAL peak position
    extraction_info.update({
        'pulse_duration_us': pulse_duration_us,
        'peak_position_us': actual_peak_position_in_pulse * dt * 1e6,
        'sampling_frequency_mhz': fs / 1e6,
        'samples_extracted': len(pulse_signal),
        'target_peak_position_us': peak_position_us,
        'pre_peak_samples': pre_peak_samples,
        'post_peak_samples': post_peak_samples
    })

    return pulse_signal, pulse_time, extraction_info


def visualize_pulse_extraction(original_signal, original_time, pulse_signal, pulse_time,
                              peak_indices, peak_times, extraction_info, filename=""):
    """
    Create comprehensive visualization of pulse extraction process.

    Parameters:
    ----------
    original_signal : array-like
        Original full signal
    original_time : array-like
        Original time vector
    pulse_signal : array-like
        Extracted pulse signal
    pulse_time : array-like
        Extracted pulse time vector
    peak_indices : array-like
        Detected peak indices in original signal
    peak_times : array-like
        Detected peak times
    extraction_info : dict
        Information about the extraction process
    filename : str
        Original filename for plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Original signal with detected peaks
    ax1.plot(original_time * 1e6, original_signal, 'b-', linewidth=0.8, alpha=0.7, label='Original Signal')
    ax1.plot(peak_times * 1e6, original_signal[peak_indices], 'ro', markersize=6,
             label=f'Detected Peaks ({len(peak_indices)})')

    # Highlight the extraction region if peak is within bounds
    if extraction_info['boundary_condition'] == 'normal':
        extraction_start = pulse_time[0] * 1e6
        extraction_end = pulse_time[-1] * 1e6
        ax1.axvspan(extraction_start, extraction_end, alpha=0.2, color='red',
                   label='Extraction Region')

    ax1.set_xlabel('Time (µs)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Amplitude (V)', fontsize=11, fontweight='bold')
    ax1.set_title(f'PD Signal Analysis - {filename}\nPeak Detection and Extraction Overview',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add signal statistics
    signal_stats = (f'Signal: {len(original_signal)} samples, '
                   f'{(original_time[-1] - original_time[0])*1e6:.1f} µs, '
                   f'Peaks: {len(peak_indices)}')
    ax1.text(0.02, 0.98, signal_stats, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: Extracted pulse with peak marker
    ax2.plot(pulse_time * 1e6, pulse_signal, 'g-', linewidth=2, label='Extracted Pulse')

    # Mark the peak position
    peak_pos_in_pulse = extraction_info['peak_position_in_pulse']
    if peak_pos_in_pulse < len(pulse_signal):
        peak_time_in_pulse = pulse_time[peak_pos_in_pulse] * 1e6
        peak_amp_in_pulse = pulse_signal[peak_pos_in_pulse]
        ax2.plot(peak_time_in_pulse, peak_amp_in_pulse, 'ro', markersize=8,
                label=f'Peak @ {extraction_info["peak_position_us"]:.2f} µs')

    # Mark the specified time boundaries
    if len(pulse_time) > 0:
        ax2.axvline(pulse_time[0] * 1e6 + 1.25, color='red', linestyle='--', alpha=0.7,
                   label='Target Peak Position (1.25 µs)')

    ax2.set_xlabel('Time (µs)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Amplitude (V)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Extracted PD Pulse - 1024 samples (6.4 µs duration)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add extraction statistics
    extraction_stats = (f'Duration: {extraction_info["pulse_duration_us"]:.2f} µs, '
                       f'Samples: {extraction_info["samples_extracted"]}, '
                       f'Peak @ {extraction_info["peak_position_us"]:.2f} µs')
    ax2.text(0.02, 0.98, extraction_stats, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Add boundary condition info if applicable
    if extraction_info['boundary_condition'] != 'normal':
        boundary_info = f"Boundary: {extraction_info['boundary_condition']}, Padding: {extraction_info['padding_samples']} samples"
        ax2.text(0.02, 0.02, boundary_info, transform=ax2.transAxes,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    return fig


def visualize_all_pulses(original_signal, original_time, all_pulse_data,
                        peak_indices, peak_times, filename=""):
    """
    Create comprehensive visualization of all extracted pulses.

    Parameters:
    ----------
    original_signal : array-like
        Original full signal
    original_time : array-like
        Original time vector
    all_pulse_data : list
        List of dictionaries containing pulse data for each peak
    peak_indices : array-like
        Detected peak indices in original signal
    peak_times : array-like
        Detected peak times
    filename : str
        Original filename for plot title
    """
    num_pulses = len(all_pulse_data)

    # Calculate grid layout
    if num_pulses <= 4:
        rows, cols = 2, 2
    elif num_pulses <= 6:
        rows, cols = 2, 3
    elif num_pulses <= 9:
        rows, cols = 3, 3
    elif num_pulses <= 12:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4

    # Create figure with original signal and all pulses
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Original signal with all detected peaks (top plot - spans full width)
    ax_main = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)
    ax_main.plot(original_time * 1e6, original_signal, 'b-', linewidth=0.8, alpha=0.7, label='Original Signal')
    ax_main.plot(peak_times * 1e6, original_signal[peak_indices], 'ro', markersize=6,
                 label=f'Detected Peaks ({len(peak_indices)})')

    # Mark peak positions with cyan color instead of multiple colors
    for i, (peak_time, pulse_data) in enumerate(zip(peak_times, all_pulse_data)):
        ax_main.axvline(peak_time * 1e6, color='cyan', linestyle='--', alpha=0.6,
                       label='Peak Positions' if i == 0 else '')

    ax_main.set_xlabel('Time (µs)', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Amplitude (V)', fontsize=11, fontweight='bold')
    ax_main.set_title(f'PD Signal Analysis - {filename}\nAll Detected Peaks and Pulse Extraction',
                     fontsize=12, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(ncol=3)

    # Add signal statistics
    signal_stats = (f'Signal: {len(original_signal)} samples, '
                   f'{(original_time[-1] - original_time[0])*1e6:.1f} µs, '
                   f'Total Peaks: {len(peak_indices)}')
    ax_main.text(0.02, 0.98, signal_stats, transform=ax_main.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot individual pulses in grid - ALL IN CYAN COLOR
    for i, pulse_data in enumerate(all_pulse_data[:rows*cols]):
        row = (i // cols) + 1
        col = i % cols
        ax = plt.subplot2grid((rows + 1, cols), (row, col))

        pulse_signal = pulse_data['pulse_signal']
        pulse_time = pulse_data['pulse_time']
        extraction_info = pulse_data['extraction_info']

        # Plot pulse in cyan color (xanh nước biển)
        ax.plot(pulse_time * 1e6, pulse_signal, 'cyan', linewidth=2,
                label=f'Pulse {i+1}')

        # Mark peak position
        peak_pos_in_pulse = extraction_info['peak_position_in_pulse']
        if peak_pos_in_pulse < len(pulse_signal):
            peak_time_in_pulse = pulse_time[peak_pos_in_pulse] * 1e6
            peak_amp_in_pulse = pulse_signal[peak_pos_in_pulse]
            ax.plot(peak_time_in_pulse, peak_amp_in_pulse, 'ro', markersize=6)

        # Mark target peak position
        if len(pulse_time) > 0:
            ax.axvline(pulse_time[0] * 1e6 + 1.25, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Time (µs)', fontsize=9)
        ax.set_ylabel('Amplitude (V)', fontsize=9)
        ax.set_title(f'Pulse {i+1} - Peak @ {peak_times[i]*1e6:.1f}µs', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add pulse info
        pulse_info = f'Amp: {pulse_data["peak_amplitude"]:.3f}V'
        ax.text(0.05, 0.95, pulse_info, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def extract_all_pulses(signal_data, time_data, peak_indices, peak_times, peak_amplitudes,
                      pulse_duration_us=6.4, peak_position_us=1.25, max_pulses=12):
    """
    Extract pulses around all detected peaks.

    Parameters:
    ----------
    signal_data : array-like
        Input signal data
    time_data : array-like
        Corresponding time vector
    peak_indices : array-like
        Indices of all detected peaks
    peak_times : array-like
        Times of all detected peaks
    peak_amplitudes : array-like
        Amplitudes of all detected peaks
    pulse_duration_us : float
        Total pulse duration in microseconds
    peak_position_us : float
        Position of peak within extracted pulse in microseconds
    max_pulses : int
        Maximum number of pulses to extract (default: 12)

    Returns:
    -------
    list
        List of dictionaries containing pulse data for each peak
    """
    all_pulse_data = []

    # Sort peaks by amplitude (strongest first) but limit to max_pulses
    sorted_indices = np.argsort(np.abs(peak_amplitudes))[::-1]
    selected_indices = sorted_indices[:min(max_pulses, len(peak_indices))]

    print(f"\nExtracting {len(selected_indices)} strongest pulses:")

    for i, idx in enumerate(selected_indices):
        peak_index = peak_indices[idx]
        peak_time = peak_times[idx]
        peak_amplitude = peak_amplitudes[idx]

        print(f"├── Pulse {i+1}: Index {peak_index}, Time {peak_time*1e6:.2f}µs, Amp {peak_amplitude:.6f}V")

        # Extract pulse around this peak
        pulse_signal, pulse_time, extraction_info = extract_pulse_around_peak(
            signal_data, time_data, peak_index,
            pulse_duration_us=pulse_duration_us,
            peak_position_us=peak_position_us
        )

        # Store pulse data
        pulse_data = {
            'pulse_signal': pulse_signal,
            'pulse_time': pulse_time,
            'extraction_info': extraction_info,
            'peak_index': peak_index,
            'peak_time': peak_time,
            'peak_amplitude': peak_amplitude,
            'pulse_number': i + 1
        }

        all_pulse_data.append(pulse_data)

    print(f"└── Total pulses extracted: {len(all_pulse_data)}")

    return all_pulse_data


def save_individual_pulse_plots(all_pulse_data, original_filename, output_folder="pulse_output", pd_type="surface"):
    """
    Save each pulse as a separate plot with red signal color.

    Parameters:
    ----------
    all_pulse_data : list
        List of dictionaries containing pulse data for each peak
    original_filename : str
        Original filename for naming convention
    output_folder : str
        Output directory for plots
    pd_type : str
        Type of partial discharge (e.g., 'surface', 'corona', 'void')
    """
    # Extract number from filename (e.g., "surface1.mat" -> "1")
    base_name = os.path.splitext(original_filename)[0]  # Remove .mat extension

    # Try to extract number from filename
    import re
    numbers = re.findall(r'\d+', base_name)
    if numbers:
        file_number = numbers[-1]  # Take the last number found
    else:
        file_number = "1"  # Default if no number found

    print(f"\nSaving individual pulse plots for {base_name}...")
    print(f"PD Type: {pd_type}")
    print(f"File number detected: {file_number}")

    for i, pulse_data in enumerate(all_pulse_data):
        pulse_signal = pulse_data['pulse_signal']
        pulse_time = pulse_data['pulse_time']
        extraction_info = pulse_data['extraction_info']

        # Create individual pulse plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot pulse in red as requested
        ax.plot(pulse_time * 1e6, pulse_signal, 'r-', linewidth=2,
                label=f'Pulse {i+1}')

        # Mark peak position
        peak_pos_in_pulse = extraction_info['peak_position_in_pulse']
        peak_time_in_pulse = extraction_info['peak_position_us']  # Get from extraction_info

        if peak_pos_in_pulse < len(pulse_signal):
            peak_amp_in_pulse = pulse_signal[peak_pos_in_pulse]
            ax.plot(peak_time_in_pulse, peak_amp_in_pulse, 'ko', markersize=8,
                   label=f'Peak @ {peak_time_in_pulse:.2f} µs')

        # Mark target peak position (1.25 µs from start)
        ax.axvline(1.25, color='black', linestyle='--', alpha=0.5, linewidth=1,
                   label='Target Peak (1.25 µs)')

        # ★ SET STANDARDIZED TIME AXIS: 0 to 6.4 µs with clear ticks
        ax.set_xlim(0, 6.4)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])

        ax.set_xlabel('Time (µs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
        ax.set_title(f'Pulse {i+1} from {base_name} ({pd_type.upper()} discharge)\n1024 samples, 6.4 µs duration',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        # Add pulse statistics
        pulse_stats = (f'PD Type: {pd_type.upper()}\n'
                      f'Peak Amplitude: {pulse_data["peak_amplitude"]:.6f} V\n'
                      f'Peak Position: {peak_time_in_pulse:.2f} µs\n'
                      f'Target: 1.25 µs\n'
                      f'Samples: {extraction_info["samples_extracted"]}')
        ax.text(0.02, 0.98, pulse_stats, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Save individual pulse plot with naming convention: pulse{X}_{pd_type}Y
        plot_filename = f'pulse{i+1}_{pd_type}{file_number}.png'
        plot_path = os.path.join(output_folder, plot_filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"├── Saved: {plot_filename}")

    print(f"└── Total individual plots saved: {len(all_pulse_data)}")


def save_individual_pulse_mat_files(all_pulse_data, original_filename, output_folder="pulse_output", pd_type="surface"):
    """
    Save each pulse as a separate .mat file with Voltage and Time columns.

    Parameters:
    ----------
    all_pulse_data : list
        List of dictionaries containing pulse data for each peak
    original_filename : str
        Original filename for naming convention
    output_folder : str
        Output directory for .mat files
    pd_type : str
        Type of partial discharge (e.g., 'surface', 'corona', 'void')
    """
    # Extract number from filename
    base_name = os.path.splitext(original_filename)[0]

    import re
    numbers = re.findall(r'\d+', base_name)
    if numbers:
        file_number = numbers[-1]
    else:
        file_number = "1"

    print(f"\nSaving individual pulse .mat files for {base_name}...")
    print(f"PD Type: {pd_type}")
    print(f"File number detected: {file_number}")

    for i, pulse_data in enumerate(all_pulse_data):
        pulse_signal = pulse_data['pulse_signal']
        pulse_time = pulse_data['pulse_time']

        # Prepare data in the requested format: Voltage and Time columns
        mat_data = {
            'Voltage': pulse_signal.reshape(-1, 1),  # Column vector
            'Time': pulse_time.reshape(-1, 1),       # Column vector
            'pulse_info': {
                'pulse_number': i + 1,
                'original_filename': original_filename,
                'pd_type': pd_type,
                'peak_amplitude': pulse_data['peak_amplitude'],
                'peak_time': pulse_data['peak_time'],
                'peak_index': pulse_data['peak_index'],
                'extraction_info': pulse_data['extraction_info']
            }
        }

        # Save with naming convention: pulse{X}_{pd_type}{Y}.mat
        mat_filename = f'pulse{i+1}_{pd_type}{file_number}.mat'
        mat_path = os.path.join(output_folder, mat_filename)

        scipy.io.savemat(mat_path, mat_data)
        print(f"├── Saved: {mat_filename} (Voltage: {len(pulse_signal)} samples, Time: {len(pulse_time)} samples)")

    print(f"└── Total individual .mat files saved: {len(all_pulse_data)}")


def process_pd_file_for_pulse_extraction(file_path, output_folder="pulse_output",
                                        prominence_factor=0.1, visualize=True, extract_all=True, pd_type="surface"):
    """
    Process a single PD .mat file for pulse extraction.

    Parameters:
    ----------
    file_path : str
        Path to the .mat file
    output_folder : str
        Output directory for results
    prominence_factor : float
        Peak detection sensitivity (lower = more sensitive)
    visualize : bool
        Generate visualization plots
    extract_all : bool
        Extract all detected pulses (True) or only strongest (False)
    pd_type : str
        Type of partial discharge (e.g., 'surface', 'corona', 'void')

    Returns:
    -------
    dict
        Processing results and extracted pulse data
    """
    try:
        # Load .mat file
        filename = os.path.basename(file_path)
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"PD Type: {pd_type.upper()}")
        print(f"{'='*60}")

        mat_data = scipy.io.loadmat(file_path)

        # Extract signal data (same format as DWT_Scalogram_Enhanced.py)
        if 'Trace_3_VOLT_filtered_db4' in mat_data and 'Time_s' in mat_data:
            signal_data = mat_data['Trace_3_VOLT_filtered_db4'].flatten()
            time_data = mat_data['Time_s'].flatten()
        else:
            print(f"ERROR: Required data not found in {filename}")
            print(f"Available keys: {list(mat_data.keys())}")
            return None

        print(f"Loaded signal: {len(signal_data)} samples")
        print(f"Time range: {time_data[0]*1e6:.2f} to {time_data[-1]*1e6:.2f} µs")

        # Step 1: Detect peaks in the signal
        peak_indices, peak_times, peak_amplitudes, filtered_signal = detect_pd_peaks(
            signal_data, time_data, prominence_factor=prominence_factor
        )

        if len(peak_indices) == 0:
            print("No peaks detected! Try reducing prominence_factor.")
            return None

        # Create output directory
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if extract_all:
            # Step 2: Extract all detected pulses
            all_pulse_data = extract_all_pulses(
                signal_data, time_data, peak_indices, peak_times, peak_amplitudes,
                pulse_duration_us=6.4, peak_position_us=1.25
            )

            print(f"\nAll pulse extraction completed:")
            print(f"├── Total pulses extracted: {len(all_pulse_data)}")
            print(f"├── Duration per pulse: 6.4 µs (1024 samples)")
            print(f"└── Peak position: 1.25 µs")

            # Step 3: Save individual pulse plots (red color) with pd_type
            save_individual_pulse_plots(all_pulse_data, filename, output_folder, pd_type=pd_type)

            # Step 4: Save individual pulse .mat files (Voltage & Time columns) with pd_type
            save_individual_pulse_mat_files(all_pulse_data, filename, output_folder, pd_type=pd_type)

            # Step 5: Create overview visualization for all pulses
            if visualize:
                print("Generating overview visualization for all pulses...")

                fig = visualize_all_pulses(
                    signal_data, time_data, all_pulse_data,
                    peak_indices, peak_times, filename
                )

                # Save overview plot
                plot_filename = filename.replace('.mat', '_all_pulses_overview.png')
                plot_path = os.path.join(output_folder, plot_filename)
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Overview plot saved: {plot_filename}")

            return {
                'all_pulse_data': all_pulse_data,
                'original_filename': filename,
                'pd_type': pd_type,
                'total_peaks_detected': len(peak_indices),
                'total_pulses_extracted': len(all_pulse_data),
                'peak_detection_info': {
                    'peak_indices': peak_indices,
                    'peak_times': peak_times,
                    'peak_amplitudes': peak_amplitudes,
                    'prominence_factor': prominence_factor
                }
            }

        else:
            # Step 2: Select the strongest peak for extraction
            strongest_peak_idx = np.argmax(np.abs(peak_amplitudes))
            selected_peak_index = peak_indices[strongest_peak_idx]
            selected_peak_time = peak_times[strongest_peak_idx]
            selected_peak_amplitude = peak_amplitudes[strongest_peak_idx]

            print(f"\nSelected strongest peak:")
            print(f"├── Index: {selected_peak_index}")
            print(f"├── Time: {selected_peak_time*1e6:.2f} µs")
            print(f"└── Amplitude: {selected_peak_amplitude:.6f} V")

            # Step 3: Extract pulse around the selected peak
            pulse_signal, pulse_time, extraction_info = extract_pulse_around_peak(
                signal_data, time_data, selected_peak_index,
                pulse_duration_us=6.4, peak_position_us=1.25
            )

            print(f"\nPulse extraction completed:")
            print(f"├── Extracted samples: {len(pulse_signal)}")
            print(f"├── Duration: {extraction_info['pulse_duration_us']:.2f} µs")
            print(f"├── Peak position: {extraction_info['peak_position_us']:.2f} µs")
            print(f"└── Boundary condition: {extraction_info['boundary_condition']}")

            # Step 4: Create visualization
            if visualize:
                print("Generating visualization...")

                fig = visualize_pulse_extraction(
                    signal_data, time_data, pulse_signal, pulse_time,
                    peak_indices, peak_times, extraction_info, filename
                )

                # Save plot
                plot_filename = filename.replace('.mat', '_pulse_extraction.png')
                plot_path = os.path.join(output_folder, plot_filename)
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Plot saved: {plot_filename}")

            # Step 5: Save extracted pulse data
            pulse_filename = filename.replace('.mat', '_extracted_pulse.mat')
            pulse_path = os.path.join(output_folder, pulse_filename)

            # Prepare data for saving
            pulse_data = {
                'pulse_signal': pulse_signal,
                'pulse_time': pulse_time,
                'extraction_info': extraction_info,
                'original_filename': filename,
                'peak_info': {
                    'selected_peak_index': selected_peak_index,
                    'selected_peak_time': selected_peak_time,
                    'selected_peak_amplitude': selected_peak_amplitude,
                    'total_peaks_detected': len(peak_indices)
                }
            }

            scipy.io.savemat(pulse_path, pulse_data)
            print(f"Pulse data saved: {pulse_filename}")

            return pulse_data

    except Exception as e:
        print(f"ERROR processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def batch_process_pd_files(input_folder, output_folder="pulse_output",
                          prominence_factor=0.1, max_files=None, pd_type="surface"):
    """
    Batch process multiple PD files for pulse extraction.

    Parameters:
    ----------
    input_folder : str
        Directory containing .mat files
    output_folder : str
        Output directory for results
    prominence_factor : float
        Peak detection sensitivity
    max_files : int, optional
        Maximum number of files to process (None = all files)
    pd_type : str
        Type of partial discharge (e.g., 'surface', 'corona', 'void')
    """
    print("="*80)
    print("Partial Discharge Pulse Extraction - Batch Processing")
    print("="*80)

    if not os.path.exists(input_folder):
        print(f"ERROR: Input folder '{input_folder}' does not exist!")
        return

    # Get .mat files
    mat_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.mat')])

    if not mat_files:
        print(f"ERROR: No .mat files found in '{input_folder}'")
        return

    if max_files:
        mat_files = mat_files[:max_files]

    print(f"Found {len(mat_files)} .mat files to process")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"PD Type: {pd_type.upper()}")
    print(f"Peak detection sensitivity: {prominence_factor}")
    print(f"Pulse parameters: 1024 samples, 6.4µs duration, peak @ 1.25µs")

    # Create output directory
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process files
    successful_extractions = 0
    failed_extractions = 0

    for i, filename in enumerate(mat_files):
        file_path = os.path.join(input_folder, filename)

        print(f"\n[{i+1}/{len(mat_files)}] Processing: {filename}")

        result = process_pd_file_for_pulse_extraction(
            file_path, output_folder, prominence_factor=prominence_factor,
            pd_type=pd_type
        )

        if result:
            successful_extractions += 1
        else:
            failed_extractions += 1

    # Summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(mat_files)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Output directory: {output_folder}")
    print(f"{'='*80}")


def main():
    """
    Main function for PD pulse extraction with user interaction.
    """
    print("="*80)
    print("Partial Discharge Pulse Detection and Extraction")
    print("="*80)
    print("Extract 1024-sample pulses (6.4µs) with peak positioned at 1.25µs")
    print("Each pulse saved as individual red plot and .mat file with Voltage/Time columns")
    print("Naming format: pulseX_{pd_type}Y.png/mat (X=pulse number, Y=file number)")
    print()

    # Get user input
    print("Choose processing mode:")
    print("1. Single file")
    print("2. Batch process folder")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        # Single file mode
        file_path = input("Enter .mat file path: ").strip()

        # Remove quotes if user copied path with quotes
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]

        if not os.path.exists(file_path):
            print(f"ERROR: File '{file_path}' does not exist!")
            return

        # Ask for PD type
        pd_type = input("Enter PD type (e.g., surface, corona, void, default: surface): ").strip().lower()
        if not pd_type:
            pd_type = "surface"

        prominence = input("Peak detection sensitivity (0.1-1.0, default 0.3): ").strip()
        try:
            prominence_factor = float(prominence) if prominence else 0.3
        except:
            prominence_factor = 0.3

        extract_mode = input("Extract all pulses? (y/n, default y): ").strip().lower()
        extract_all = extract_mode != 'n'

        output_folder = input("Output folder (default: pulse_output): ").strip()
        if not output_folder:
            output_folder = "pulse_output"

        print(f"\nProcessing single file: {os.path.basename(file_path)}")
        print(f"PD Type: {pd_type.upper()}")
        print(f"Extract all pulses: {extract_all}")
        print(f"Peak sensitivity: {prominence_factor}")
        print(f"Output folder: {output_folder}")

        result = process_pd_file_for_pulse_extraction(
            file_path,
            output_folder,
            prominence_factor=prominence_factor,
            extract_all=extract_all,
            pd_type=pd_type
        )

        if result:
            print("\n" + "="*60)
            print("✓ SINGLE FILE PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            if extract_all and 'total_pulses_extracted' in result:
                print(f"Total peaks detected: {result['total_peaks_detected']}")
                print(f"Total pulses extracted: {result['total_pulses_extracted']}")
                print(f"Individual files created:")
                print(f"├── Red plots: pulse1_{pd_type}*.png, pulse2_{pd_type}*.png, ...")
                print(f"├── Mat files: pulse1_{pd_type}*.mat, pulse2_{pd_type}*.mat, ...")
                print(f"└── Overview: *_all_pulses_overview.png")
            else:
                print("Single strongest pulse extracted and visualized")
            print(f"Output location: {output_folder}")
        else:
            print("\n✗ Single file processing failed!")

    elif choice == "2":
        # Batch mode
        input_folder = input("Enter input folder path: ").strip()

        # Remove quotes if user copied path with quotes
        if input_folder.startswith('"') and input_folder.endswith('"'):
            input_folder = input_folder[1:-1]

        if not os.path.exists(input_folder):
            print(f"ERROR: Folder '{input_folder}' does not exist!")
            return

        # Ask for PD type
        pd_type = input("Enter PD type (e.g., surface, corona, void, default: surface): ").strip().lower()
        if not pd_type:
            pd_type = "surface"

        prominence = input("Peak detection sensitivity (0.1-1.0, default 0.3): ").strip()
        try:
            prominence_factor = float(prominence) if prominence else 0.3
        except:
            prominence_factor = 0.3

        max_files_input = input("Max files to process (default: all): ").strip()
        try:
            max_files = int(max_files_input) if max_files_input else None
        except:
            max_files = None

        output_folder = input("Output folder (default: pulse_output): ").strip()
        if not output_folder:
            output_folder = "pulse_output"

        print(f"\nBatch processing folder: {input_folder}")
        print(f"PD Type: {pd_type.upper()}")
        print(f"Peak sensitivity: {prominence_factor}")
        print(f"Max files: {max_files if max_files else 'all'}")
        print(f"Output folder: {output_folder}")

        batch_process_pd_files(
            input_folder,
            output_folder,
            prominence_factor=prominence_factor,
            max_files=max_files,
            pd_type=pd_type
        )

    else:
        print("Invalid choice! Please select 1 or 2.")


if __name__ == "__main__":
    main()
