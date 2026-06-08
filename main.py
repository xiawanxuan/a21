import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Astronomical Spectrum Redshift Measurement and Classification System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    generate_parser = subparsers.add_parser("generate", help="Generate sample dataset")
    generate_parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of samples to generate"
    )
    generate_parser.add_argument(
        "--output", type=str, default=None, help="Output file path"
    )

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data_path", type=str, default=None, help="Path to training data"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size"
    )
    train_parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )

    test_parser = subparsers.add_parser("test", help="Evaluate the model")
    test_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    test_parser.add_argument(
        "--data_path", type=str, default=None, help="Path to test data"
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference on spectra")
    predict_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--input", type=str, required=True, help="Path to input spectra file (.npy or .npz)"
    )
    predict_parser.add_argument(
        "--output", type=str, default=None, help="Output file path"
    )
    predict_parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )

    visualize_parser = subparsers.add_parser("visualize", help="Generate visualization from test results")
    visualize_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    visualize_parser.add_argument(
        "--data_path", type=str, default=None, help="Path to test data"
    )
    visualize_parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for plots"
    )

    args = parser.parse_args()

    config = Config()
    set_seed(config.seed)

    if args.command == "generate":
        from scripts.generate_sample_data import generate_dataset

        print("Generating sample dataset...")
        generate_dataset(
            num_samples=args.num_samples,
            config=config,
            save_path=args.output,
        )

    elif args.command == "train":
        from src.data_processing import create_dataloaders
        from src.models import build_model
        from src.train_eval import train, load_checkpoint, evaluate

        if args.epochs is not None:
            config.num_epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.lr is not None:
            config.learning_rate = args.lr

        print(f"Loading data from {args.data_path or config.data_dir + '/spectrum_data.npz'}...")
        train_loader, val_loader, test_loader = create_dataloaders(config, args.data_path)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

        model = build_model(config)

        if args.resume:
            optimizer = None
            load_checkpoint(model, optimizer, args.resume, config.device)

        model, history = train(config, train_loader, val_loader, model)

        print("\n--- Final Test Evaluation ---")
        best_model_path = os.path.join(config.model_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            from src.train_eval import load_checkpoint
            load_checkpoint(model, None, best_model_path, config.device)
        test_metrics = evaluate(model, test_loader, config)

        result_path = os.path.join(config.result_dir, "test_results.json")
        os.makedirs(config.result_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test results saved to {result_path}")

    elif args.command == "test":
        from src.data_processing import create_dataloaders
        from src.inference_visualization import load_model
        from src.train_eval import evaluate

        print(f"Loading model from {args.checkpoint}...")
        model = load_model(config, args.checkpoint)

        print(f"Loading test data...")
        _, _, test_loader = create_dataloaders(config, args.data_path)

        print("Running evaluation...")
        test_metrics = evaluate(model, test_loader, config)

        result_path = os.path.join(config.result_dir, "test_results.json")
        os.makedirs(config.result_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Test results saved to {result_path}")

    elif args.command == "predict":
        from src.inference_visualization import load_model, predict_batch, generate_all_plots
        from src.data_processing import create_dataloaders

        print(f"Loading model from {args.checkpoint}...")
        model = load_model(config, args.checkpoint)

        print(f"Loading input data from {args.input}...")
        if args.input.endswith(".npy"):
            spectra = np.load(args.input)
        elif args.input.endswith(".npz"):
            data = np.load(args.input)
            spectra = data["spectra"]
        else:
            raise ValueError("Input file must be .npy or .npz format")

        print(f"Running inference on {len(spectra)} spectra...")
        results = predict_batch(model, spectra, config)

        if args.output is None:
            args.output = os.path.join(config.output_dir, "predictions.json")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Predictions saved to {args.output}")

        if args.visualize:
            print("Generating visualization plots...")
            from src.data_processing import SpectrumDataset, DataLoader
            from src.inference_visualization import generate_all_plots

            if "redshifts" in data and "labels" in data:
                test_dataset = SpectrumDataset(
                    spectra, data["redshifts"], data["labels"], config, augment=False
                )
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                generate_all_plots(model, test_loader, config)
            else:
                print("Ground truth labels not available, skipping comparison plots")

    elif args.command == "visualize":
        from src.data_processing import create_dataloaders
        from src.inference_visualization import load_model, generate_all_plots

        print(f"Loading model from {args.checkpoint}...")
        model = load_model(config, args.checkpoint)

        print(f"Loading test data...")
        _, _, test_loader = create_dataloaders(config, args.data_path)

        output_dir = args.output_dir or config.output_dir
        print(f"Generating visualization plots to {output_dir}...")
        generate_all_plots(model, test_loader, config, output_dir=output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
