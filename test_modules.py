import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


def test_config():
    print("Testing Config...")
    config = Config()
    assert config.spectrum_length == 1024
    assert config.num_classes == 5
    assert config.d_model == 128
    print("  Config test PASSED")


def test_data_processing():
    print("Testing data_processing module...")
    from src.data_processing import (
        normalize_spectrum,
        min_max_normalize,
        add_noise,
        shift_spectrum,
        scale_spectrum,
        SpectrumDataset,
        split_data,
        create_inference_dataset,
    )

    spectrum = np.random.randn(1024).astype(np.float32)

    norm_spec = normalize_spectrum(spectrum)
    assert norm_spec.shape == spectrum.shape
    assert abs(np.mean(norm_spec)) < 1e-6

    mm_spec = min_max_normalize(spectrum)
    assert mm_spec.shape == spectrum.shape
    assert mm_spec.min() >= 0 and mm_spec.max() <= 1.0

    noisy_spec = add_noise(spectrum, noise_level=0.05)
    assert noisy_spec.shape == spectrum.shape

    shifted_spec = shift_spectrum(spectrum, max_shift=10)
    assert shifted_spec.shape == spectrum.shape

    scaled_spec = scale_spectrum(spectrum, scale_range=(0.9, 1.1))
    assert scaled_spec.shape == spectrum.shape

    num_samples = 100
    spectra = np.random.randn(num_samples, 1024).astype(np.float32)
    redshifts = np.random.rand(num_samples).astype(np.float32) * 2
    labels = np.random.randint(0, 5, num_samples).astype(np.int64)

    config = Config()
    dataset = SpectrumDataset(spectra, redshifts, labels, config, augment=True)
    assert len(dataset) == num_samples
    spec, r, l = dataset[0]
    assert spec.shape == torch.Size([1, 1024])
    assert r.shape == torch.Size([1])
    assert l.shape == torch.Size([1])

    train_data, val_data, test_data = split_data(spectra, redshifts, labels, config)
    assert len(train_data[0]) + len(val_data[0]) + len(test_data[0]) == num_samples

    infer_tensor = create_inference_dataset(spectra, config)
    assert infer_tensor.shape == torch.Size([num_samples, 1, 1024])

    print("  data_processing test PASSED")


def test_models():
    print("Testing models module...")
    from src.models import (
        LearnablePositionalEmbedding,
        SinusoidalPositionalEncoding,
        MultiScaleSpectrumEmbedding,
        TransformerEncoder,
        RedshiftRegressionHead,
        ClassificationHead,
        SpectrumTransformer,
        build_model,
        count_parameters,
    )

    config = Config()

    model = build_model(config)
    total_params, trainable_params = count_parameters(model)
    assert total_params > 0
    assert trainable_params > 0

    batch_size = 4
    x = torch.randn(batch_size, 1, config.spectrum_length)
    redshift_pred, class_pred = model(x)
    assert redshift_pred.shape == torch.Size([batch_size, 1])
    assert class_pred.shape == torch.Size([batch_size, config.num_classes])

    features = model.extract_features(x)
    assert features.shape[0] == batch_size
    assert features.shape[2] == config.d_model

    print("  models test PASSED")


def test_train_eval():
    print("Testing train_eval module...")
    from src.train_eval import CombinedLoss, compute_metrics

    config = Config()
    criterion = CombinedLoss(config)

    batch_size = 8
    redshift_pred = torch.randn(batch_size, 1)
    redshift_true = torch.randn(batch_size, 1)
    class_pred = torch.randn(batch_size, config.num_classes)
    class_true = torch.randint(0, config.num_classes, (batch_size, 1))

    loss, r_loss, c_loss = criterion(redshift_pred, redshift_true, class_pred, class_true)
    assert loss.item() > 0
    assert r_loss.item() > 0
    assert c_loss.item() > 0

    metrics = compute_metrics(redshift_pred, redshift_true, class_pred, class_true)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "sigma_ni" in metrics
    assert "outlier_rate" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    print("  train_eval test PASSED")


def test_inference_visualization():
    print("Testing inference_visualization module...")
    from src.inference_visualization import (
        plot_spectrum,
        plot_redshift_comparison,
        plot_residuals,
        plot_confusion_matrix,
        plot_training_history,
    )

    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    spectrum = np.random.randn(1024)
    plot_spectrum(spectrum, title="Test Spectrum", save_path=os.path.join(config.output_dir, "test_spectrum.png"))
    assert os.path.exists(os.path.join(config.output_dir, "test_spectrum.png"))

    true_z = np.random.rand(100) * 2
    pred_z = true_z + np.random.randn(100) * 0.1
    plot_redshift_comparison(true_z, pred_z, save_path=os.path.join(config.output_dir, "test_z_comparison.png"))
    assert os.path.exists(os.path.join(config.output_dir, "test_z_comparison.png"))

    plot_residuals(true_z, pred_z, save_path=os.path.join(config.output_dir, "test_residuals.png"))
    assert os.path.exists(os.path.join(config.output_dir, "test_residuals.png"))

    true_labels = np.random.randint(0, 5, 100)
    pred_labels = np.random.randint(0, 5, 100)
    plot_confusion_matrix(true_labels, pred_labels, config.class_names, save_path=os.path.join(config.output_dir, "test_cm.png"))
    assert os.path.exists(os.path.join(config.output_dir, "test_cm.png"))

    history = {
        "train": [{"total_loss": 1.0, "mae": 0.5, "accuracy": 0.7, "sigma_ni": 0.05} for _ in range(5)],
        "val": [{"total_loss": 1.2, "mae": 0.6, "accuracy": 0.65, "sigma_ni": 0.06} for _ in range(5)],
    }
    plot_training_history(history, save_path=os.path.join(config.output_dir, "test_history.png"))
    assert os.path.exists(os.path.join(config.output_dir, "test_history.png"))

    print("  inference_visualization test PASSED")


def test_prediction():
    print("Testing prediction functionality...")
    from src.models import build_model
    from src.inference_visualization import predict_single, predict_batch

    config = Config()
    model = build_model(config)
    model.eval()

    spectrum = np.random.randn(config.spectrum_length).astype(np.float32)
    result = predict_single(model, spectrum, config, device=torch.device("cpu"))
    assert "redshift" in result
    assert "class_index" in result
    assert "class_name" in result
    assert "class_probabilities" in result
    assert len(result["class_probabilities"]) == config.num_classes

    num_samples = 20
    spectra = np.random.randn(num_samples, config.spectrum_length).astype(np.float32)
    results = predict_batch(model, spectra, config, device=torch.device("cpu"), batch_size=8)
    assert len(results) == num_samples
    assert "redshift" in results[0]

    print("  prediction test PASSED")


def test_spectral_analysis():
    print("Testing spectral_analysis module...")
    from src.spectral_analysis import (
        smooth_spectrum,
        calculate_baseline,
        detect_spectral_lines,
        compute_spectral_features,
        detect_anomalous_spectrum,
        batch_detect_lines,
        batch_detect_anomalies,
    )

    config = Config()
    np.random.seed(42)

    spectrum = np.random.randn(config.spectrum_length).astype(np.float32)

    smoothed = smooth_spectrum(spectrum, window_size=5)
    assert smoothed.shape == spectrum.shape

    baseline = calculate_baseline(spectrum, poly_order=3)
    assert baseline.shape == spectrum.shape

    line_result = detect_spectral_lines(spectrum)
    assert "emission_lines" in line_result
    assert "absorption_lines" in line_result
    assert "num_emission" in line_result
    assert "num_absorption" in line_result
    assert isinstance(line_result["emission_lines"], list)
    assert isinstance(line_result["absorption_lines"], list)

    features, line_result2 = compute_spectral_features(spectrum)
    assert "mean_flux" in features
    assert "std_flux" in features
    assert "skewness" in features
    assert "kurtosis" in features
    assert "num_emission_lines" in features
    assert "num_absorption_lines" in features

    anomaly_result = detect_anomalous_spectrum(spectrum)
    assert "is_anomalous" in anomaly_result
    assert "anomaly_score" in anomaly_result
    assert "reasons" in anomaly_result
    assert isinstance(anomaly_result["is_anomalous"], bool)

    num_samples = 20
    spectra = np.random.randn(num_samples, config.spectrum_length).astype(np.float32)
    spectra[0] = np.ones(config.spectrum_length) * 100.0

    batch_line_results = batch_detect_lines(spectra)
    assert len(batch_line_results) == num_samples
    assert "emission_lines" in batch_line_results[0]

    batch_anomaly_results, ref_stats = batch_detect_anomalies(spectra)
    assert len(batch_anomaly_results) == num_samples
    assert "is_anomalous" in batch_anomaly_results[0]
    assert ref_stats is not None
    assert "std_flux" in ref_stats

    print("  spectral_analysis test PASSED")


def main():
    print("=" * 60)
    print("Running module validation tests...")
    print("=" * 60)

    try:
        test_config()
    except Exception as e:
        print(f"  Config test FAILED: {e}")
        return False

    try:
        test_data_processing()
    except Exception as e:
        print(f"  data_processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        test_models()
    except Exception as e:
        print(f"  models test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        test_train_eval()
    except Exception as e:
        print(f"  train_eval test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        test_inference_visualization()
    except Exception as e:
        print(f"  inference_visualization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        test_prediction()
    except Exception as e:
        print(f"  prediction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        test_spectral_analysis()
    except Exception as e:
        print(f"  spectral_analysis test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
