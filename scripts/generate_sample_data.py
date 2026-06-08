import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def generate_galaxy_spectrum(length=1024, redshift=0.0):
    x = np.arange(length)
    base = 1.0 + 0.3 * np.sin(2 * np.pi * x / 200)
    absorption = np.exp(-((x - 300) ** 2) / (2 * 50**2)) * 0.3
    absorption += np.exp(-((x - 500) ** 2) / (2 * 30**2)) * 0.2
    absorption += np.exp(-((x - 700) ** 2) / (2 * 80**2)) * 0.15
    spectrum = base - absorption
    return spectrum


def generate_quasar_spectrum(length=1024, redshift=0.0):
    x = np.arange(length) + 1.0
    power_law = (x / 500) ** (-1.5) * 2
    emission = np.exp(-((x - 250) ** 2) / (2 * 20**2)) * 0.8
    emission += np.exp(-((x - 450) ** 2) / (2 * 15**2)) * 0.6
    emission += np.exp(-((x - 650) ** 2) / (2 * 25**2)) * 0.5
    spectrum = power_law + emission
    return spectrum


def generate_star_spectrum(length=1024, redshift=0.0):
    x = np.arange(length)
    blackbody = np.exp(-((x - 400) ** 2) / (2 * 300**2)) * 2
    absorption_lines = 0
    for i in range(20):
        pos = np.random.randint(50, length - 50)
        depth = np.random.uniform(0.05, 0.2)
        width = np.random.uniform(5, 15)
        absorption_lines += np.exp(-((x - pos) ** 2) / (2 * width**2)) * depth
    spectrum = blackbody - absorption_lines
    return spectrum


def generate_nebula_spectrum(length=1024, redshift=0.0):
    x = np.arange(length)
    continuum = np.ones(length) * 0.3
    emission_lines = 0
    emission_positions = [150, 280, 350, 480, 550, 650, 720, 850]
    for pos in emission_positions:
        intensity = np.random.uniform(0.5, 1.5)
        width = np.random.uniform(3, 10)
        emission_lines += np.exp(-((x - pos) ** 2) / (2 * width**2)) * intensity
    spectrum = continuum + emission_lines
    return spectrum


def generate_unknown_spectrum(length=1024, redshift=0.0):
    x = np.arange(length)
    spectrum = np.random.randn(length) * 0.1
    spectrum += np.sin(2 * np.pi * x / 150) * 0.2
    spectrum += np.sin(2 * np.pi * x / 300) * 0.15
    spectrum = np.abs(spectrum) + 0.5
    return spectrum


def apply_redshift(spectrum, redshift):
    length = len(spectrum)
    x = np.arange(length)
    x_shifted = x / (1 + redshift)
    shifted_spectrum = np.interp(x_shifted, x, spectrum)
    return shifted_spectrum


def add_noise(spectrum, snr_range=(10, 50)):
    snr = np.random.uniform(*snr_range)
    spec_mean = np.mean(np.abs(spectrum)) + 1e-8
    noise_level = spec_mean / snr
    noise = np.random.normal(0, noise_level, spectrum.shape)
    return spectrum + noise


def generate_spectrum(class_idx, length=1024, redshift_range=(0, 2)):
    redshift = np.random.uniform(*redshift_range)

    generators = [
        generate_galaxy_spectrum,
        generate_quasar_spectrum,
        generate_star_spectrum,
        generate_nebula_spectrum,
        generate_unknown_spectrum,
    ]

    spectrum = generators[class_idx](length, redshift)
    spectrum = apply_redshift(spectrum, redshift)
    spectrum = add_noise(spectrum, snr_range=(15, 40))

    return spectrum, redshift, class_idx


def generate_dataset(num_samples=5000, config=None, save_path=None):
    if config is None:
        config = Config()

    np.random.seed(config.seed)

    spectra = []
    redshifts = []
    labels = []

    class_proportions = [0.3, 0.2, 0.25, 0.15, 0.1]
    class_counts = [int(num_samples * p) for p in class_proportions]
    class_counts[-1] = num_samples - sum(class_counts[:-1])

    print(f"Generating dataset with {num_samples} samples...")
    print(f"Class distribution: Galaxy={class_counts[0]}, Quasar={class_counts[1]}, "
          f"Star={class_counts[2]}, Nebula={class_counts[3]}, Unknown={class_counts[4]}")

    for class_idx in range(5):
        for i in range(class_counts[class_idx]):
            spectrum, redshift, label = generate_spectrum(
                class_idx, length=config.spectrum_length, redshift_range=(0, 3)
            )
            spectra.append(spectrum)
            redshifts.append(redshift)
            labels.append(label)

    spectra = np.array(spectra, dtype=np.float32)
    redshifts = np.array(redshifts, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    valid_mask = ~(np.isnan(spectra).any(axis=1) | np.isinf(spectra).any(axis=1))
    spectra = spectra[valid_mask]
    redshifts = redshifts[valid_mask]
    labels = labels[valid_mask]
    num_valid = len(spectra)
    if num_valid < num_samples:
        print(f"Warning: {num_samples - num_valid} invalid samples removed, {num_valid} valid samples remaining")

    indices = np.arange(num_valid)
    np.random.shuffle(indices)
    spectra = spectra[indices]
    redshifts = redshifts[indices]
    labels = labels[indices]

    if save_path is None:
        save_path = os.path.join(config.data_dir, "spectrum_data.npz")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, spectra=spectra, redshifts=redshifts, labels=labels)
    print(f"Dataset saved to {save_path}")
    print(f"Spectra shape: {spectra.shape}")
    print(f"Redshift range: [{redshifts.min():.4f}, {redshifts.max():.4f}]")

    return spectra, redshifts, labels


if __name__ == "__main__":
    config = Config()
    generate_dataset(num_samples=5000, config=config)
