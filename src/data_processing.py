import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def normalize_spectrum(spectrum):
    mean = np.mean(spectrum)
    std = np.std(spectrum) + 1e-8
    normalized = (spectrum - mean) / std
    return normalized


def min_max_normalize(spectrum):
    min_val = np.min(spectrum)
    max_val = np.max(spectrum) + 1e-8
    normalized = (spectrum - min_val) / (max_val - min_val)
    return normalized


def add_noise(spectrum, noise_level=0.05):
    noise = np.random.normal(0, noise_level, spectrum.shape)
    return spectrum + noise


def shift_spectrum(spectrum, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        shifted = np.pad(spectrum[shift:], (0, shift), mode="edge")
    elif shift < 0:
        shifted = np.pad(spectrum[:shift], (-shift, 0), mode="edge")
    else:
        shifted = spectrum.copy()
    return shifted


def scale_spectrum(spectrum, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return spectrum * scale


def augment_spectrum(spectrum, config):
    augmented = spectrum.copy()
    if np.random.rand() < config.augment_prob:
        augmented = add_noise(augmented, config.noise_level)
    if np.random.rand() < config.augment_prob:
        augmented = shift_spectrum(augmented)
    if np.random.rand() < config.augment_prob:
        augmented = scale_spectrum(augmented)
    return augmented


class SpectrumDataset(Dataset):
    def __init__(self, spectra, redshifts, labels, config, augment=False):
        self.spectra = spectra
        self.redshifts = redshifts
        self.labels = labels
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx].copy()
        redshift = self.redshifts[idx]
        label = self.labels[idx]

        if self.augment:
            spectrum = augment_spectrum(spectrum, self.config)

        spectrum = normalize_spectrum(spectrum)
        spectrum = torch.FloatTensor(spectrum).unsqueeze(0)

        redshift = torch.FloatTensor([redshift])
        label = torch.LongTensor([label])

        return spectrum, redshift, label


def load_numpy_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    spectra = data["spectra"]
    redshifts = data["redshifts"]
    labels = data["labels"]
    return spectra, redshifts, labels


def split_data(spectra, redshifts, labels, config, shuffle=True):
    num_samples = len(spectra)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.seed(config.seed)
        np.random.shuffle(indices)

    train_end = int(num_samples * config.train_ratio)
    val_end = train_end + int(num_samples * config.val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_data = (spectra[train_idx], redshifts[train_idx], labels[train_idx])
    val_data = (spectra[val_idx], redshifts[val_idx], labels[val_idx])
    test_data = (spectra[test_idx], redshifts[test_idx], labels[test_idx])

    return train_data, val_data, test_data


def create_dataloaders(config, data_path=None):
    if data_path is None:
        data_path = os.path.join(config.data_dir, "spectrum_data.npz")

    spectra, redshifts, labels = load_numpy_data(data_path)

    train_data, val_data, test_data = split_data(spectra, redshifts, labels, config)

    train_dataset = SpectrumDataset(*train_data, config, augment=True)
    val_dataset = SpectrumDataset(*val_data, config, augment=False)
    test_dataset = SpectrumDataset(*test_data, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def create_inference_dataset(spectra, config):
    normalized_spectra = []
    for spec in spectra:
        normalized = normalize_spectrum(spec)
        normalized_spectra.append(normalized)
    normalized_spectra = np.array(normalized_spectra)
    spectra_tensor = torch.FloatTensor(normalized_spectra).unsqueeze(1)
    return spectra_tensor
