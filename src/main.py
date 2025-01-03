"""Main entry point for the ML project."""

from pathlib import Path

from src.config import config
from src.training.evaluate import evaluate_model
from src.training.train import train_model


def main() -> None:
    """Main entry point that trains and evaluates the model."""
    cfg = config()

    # Get paths from config
    model_dir = Path(cfg.model.dir) / cfg.model.version
    data_dir = Path(cfg.data.dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Train the model
    print("Training model...")
    train_model(
        data_dir=train_dir,
        model_dir=model_dir,
    )

    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(
        model_path=model_dir / "model",
        test_data_path=test_dir,
    )
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    main()
