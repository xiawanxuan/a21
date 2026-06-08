import numpy as np
from scipy.signal import find_peaks
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def smooth_spectrum(spectrum, window_size=5):
    if window_size < 2:
        return spectrum.copy()
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(spectrum, kernel, mode="same")
    return smoothed


def calculate_baseline(spectrum, poly_order=3):
    x = np.arange(len(spectrum))
    coeffs = np.polyfit(x, spectrum, poly_order)
    baseline = np.polyval(coeffs, x)
    return baseline


def normalize_spectrum_local(spectrum):
    mean = np.mean(spectrum)
    std = np.std(spectrum) + 1e-8
    return (spectrum - mean) / std


def detect_spectral_lines(spectrum, wavelength=None, height_threshold=2.0, distance=10,
                          min_prominence=0.5, window_size=5):
    if wavelength is None:
        wavelength = np.arange(len(spectrum))

    smoothed = smooth_spectrum(spectrum, window_size=window_size)
    baseline = calculate_baseline(smoothed, poly_order=2)
    continuum_removed = smoothed - baseline

    std = np.std(continuum_removed) + 1e-8

    emission_peaks, emission_props = find_peaks(
        continuum_removed,
        height=height_threshold * std,
        distance=distance,
        prominence=min_prominence * std,
    )

    absorption_peaks, absorption_props = find_peaks(
        -continuum_removed,
        height=height_threshold * std,
        distance=distance,
        prominence=min_prominence * std,
    )

    emission_lines = []
    for i, peak_idx in enumerate(emission_peaks):
        emission_lines.append({
            "index": int(peak_idx),
            "wavelength": float(wavelength[peak_idx]),
            "type": "emission",
            "intensity": float(continuum_removed[peak_idx]),
            "prominence": float(emission_props["prominences"][i]),
        })

    absorption_lines = []
    for i, peak_idx in enumerate(absorption_peaks):
        absorption_lines.append({
            "index": int(peak_idx),
            "wavelength": float(wavelength[peak_idx]),
            "type": "absorption",
            "intensity": float(continuum_removed[peak_idx]),
            "prominence": float(absorption_props["prominences"][i]),
        })

    emission_lines.sort(key=lambda x: x["prominence"], reverse=True)
    absorption_lines.sort(key=lambda x: x["prominence"], reverse=True)

    return {
        "emission_lines": emission_lines,
        "absorption_lines": absorption_lines,
        "baseline": baseline.tolist(),
        "smoothed_spectrum": smoothed.tolist(),
        "continuum_removed": continuum_removed.tolist(),
        "num_emission": len(emission_lines),
        "num_absorption": len(absorption_lines),
    }


def compute_spectral_features(spectrum):
    features = {}

    features["mean_flux"] = float(np.mean(spectrum))
    features["std_flux"] = float(np.std(spectrum))
    features["min_flux"] = float(np.min(spectrum))
    features["max_flux"] = float(np.max(spectrum))
    features["median_flux"] = float(np.median(spectrum))
    features["flux_range"] = float(np.max(spectrum) - np.min(spectrum))

    diff = np.diff(spectrum)
    features["mean_gradient"] = float(np.mean(np.abs(diff)))
    features["max_gradient"] = float(np.max(np.abs(diff)))

    spectrum_norm = normalize_spectrum_local(spectrum)
    skewness = float(np.mean(((spectrum_norm - np.mean(spectrum_norm)) ** 3)))
    kurtosis = float(np.mean(((spectrum_norm - np.mean(spectrum_norm)) ** 4)) - 3)
    features["skewness"] = skewness
    features["kurtosis"] = kurtosis

    fft = np.fft.fft(spectrum)
    power = np.abs(fft) ** 2
    features["total_power"] = float(np.sum(power))
    features["low_freq_ratio"] = float(np.sum(power[:len(power)//10]) / (np.sum(power) + 1e-8))

    line_results = detect_spectral_lines(spectrum)
    features["num_emission_lines"] = line_results["num_emission"]
    features["num_absorption_lines"] = line_results["num_absorption"]
    features["total_line_count"] = line_results["num_emission"] + line_results["num_absorption"]

    return features, line_results


def detect_anomalous_spectrum(spectrum, features=None, threshold=3.0, reference_stats=None):
    if features is None:
        features, _ = compute_spectral_features(spectrum)

    anomaly_scores = {}
    is_anomalous = False
    reasons = []

    if reference_stats is not None:
        for key in ["std_flux", "skewness", "kurtosis", "max_gradient", "total_line_count"]:
            if key in reference_stats and key in features:
                mean_val = reference_stats[key]["mean"]
                std_val = reference_stats[key]["std"] + 1e-8
                z_score = abs(features[key] - mean_val) / std_val
                anomaly_scores[key] = float(z_score)
                if z_score > threshold:
                    is_anomalous = True
                    reasons.append(f"{key} z-score={z_score:.2f} exceeds threshold")
    else:
        if features["std_flux"] < 1e-6:
            is_anomalous = True
            reasons.append("Near-zero variance")

        if abs(features["skewness"]) > 5.0:
            is_anomalous = True
            reasons.append(f"Extreme skewness: {features['skewness']:.2f}")

        if features["kurtosis"] > 20:
            is_anomalous = True
            reasons.append(f"Extreme kurtosis: {features['kurtosis']:.2f}")

        if features["max_gradient"] > 10 * features["std_flux"]:
            is_anomalous = True
            reasons.append("Abnormally sharp features")

        if features["num_emission_lines"] > 50 or features["num_absorption_lines"] > 50:
            is_anomalous = True
            reasons.append("Too many spectral lines")

        if features["total_line_count"] == 0 and features["std_flux"] < 0.1:
            is_anomalous = True
            reasons.append("Featureless spectrum with low variance")

    overall_score = float(np.mean(list(anomaly_scores.values()))) if anomaly_scores else 0.0

    return {
        "is_anomalous": is_anomalous,
        "anomaly_score": overall_score,
        "reasons": reasons,
        "anomaly_scores": anomaly_scores,
    }


def batch_detect_lines(spectra, wavelength=None, **kwargs):
    all_results = []
    for spectrum in spectra:
        result = detect_spectral_lines(spectrum, wavelength=wavelength, **kwargs)
        all_results.append(result)
    return all_results


def batch_detect_anomalies(spectra, reference_stats=None, threshold=3.0):
    all_features = []
    all_results = []

    for spectrum in spectra:
        features, line_result = compute_spectral_features(spectrum)
        all_features.append(features)

    if reference_stats is None:
        ref = {}
        feature_keys = all_features[0].keys()
        for key in feature_keys:
            values = [f[key] for f in all_features if isinstance(f[key], (int, float))]
            if values:
                ref[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        reference_stats = ref

    for features in all_features:
        result = detect_anomalous_spectrum(
            None, features=features, threshold=threshold, reference_stats=reference_stats
        )
        all_results.append(result)

    return all_results, reference_stats
