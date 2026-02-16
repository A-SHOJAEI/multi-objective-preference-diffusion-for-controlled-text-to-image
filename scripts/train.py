#!/usr/bin/env python
"""Training script for preference-guided diffusion model."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src/ to path for development (when package not installed)
project_root = Path(__file__).parent.parent
if (project_root / "src").exists():
    sys.path.insert(0, str(project_root / "src"))

try:
    from multi_objective_preference_diffusion_for_controlled_text_to_image.data import (
        ConceptualCaptionsDataset,
        PreferenceDataset,
    )
    from multi_objective_preference_diffusion_for_controlled_text_to_image.models import (
        PreferenceDiffusionModel,
    )
    from multi_objective_preference_diffusion_for_controlled_text_to_image.training import (
        PreferenceDiffusionTrainer,
    )
    from multi_objective_preference_diffusion_for_controlled_text_to_image.utils import (
        load_config,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please install the package: pip install -e .")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train preference-guided diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    return parser.parse_args()


def create_dataloaders(config, device):
    """Create training and validation data loaders.

    Args:
        config: Configuration object
        device: Training device

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating datasets...")

    # Training dataset
    train_dataset = ConceptualCaptionsDataset(
        split="train",
        max_samples=config.data.get("max_train_samples", 5000),
        image_size=config.data.get("image_size", 256),
        seed=config.training.get("seed", 42),
    )

    # Validation dataset
    val_dataset = ConceptualCaptionsDataset(
        split="validation",
        max_samples=config.data.get("max_val_samples", 1000),
        image_size=config.data.get("image_size", 256),
        seed=config.training.get("seed", 42),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.get("batch_size", 16),
        shuffle=True,
        num_workers=config.data.get("num_workers", 4),
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.get("batch_size", 16),
        shuffle=False,
        num_workers=config.data.get("num_workers", 4),
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def main() -> None:
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set random seed
        seed = config.training.get("seed", 42)
        set_seed(seed)

        # Setup device
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")

        # Create data loaders
        train_loader, val_loader = create_dataloaders(config, device)

        # Initialize model
        logger.info("Initializing model...")
        model = PreferenceDiffusionModel(
            num_objectives=config.model.get("num_objectives", 3),
            use_dynamic_guidance=config.model.get("use_dynamic_guidance", True),
            guidance_scale=config.model.get("guidance_scale", 7.5),
            num_inference_steps=config.model.get("num_inference_steps", 50),
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

        # Create optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=config.training.get("learning_rate", 0.0001),
            weight_decay=config.training.get("weight_decay", 0.01),
            betas=(
                config.training.get("adam_beta1", 0.9),
                config.training.get("adam_beta2", 0.999),
            ),
        )

        # Create learning rate scheduler
        num_epochs = config.training.get("num_epochs", 50)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config.training.get("min_learning_rate", 0.000001),
        )

        # Initialize trainer
        trainer = PreferenceDiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=config.training.get("use_mixed_precision", True),
            gradient_clip=config.training.get("gradient_clip", 1.0),
            checkpoint_dir=config.training.get("checkpoint_dir", "checkpoints"),
            log_interval=config.training.get("log_interval", 100),
        )

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # MLflow tracking (optional)
        try:
            import mlflow

            mlflow.set_experiment(config.get("experiment_name", "preference-diffusion"))
            mlflow.start_run()

            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": config.training.get("learning_rate", 0.0001),
                "batch_size": config.training.get("batch_size", 16),
                "num_epochs": num_epochs,
                "model_params": num_params,
                "seed": seed,
            })

            use_mlflow = True
            logger.info("MLflow tracking enabled")

        except ImportError:
            logger.info("MLflow not available, skipping experiment tracking")
            use_mlflow = False
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            use_mlflow = False

        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            num_epochs=num_epochs,
            early_stopping_patience=config.training.get("early_stopping_patience", 10),
            save_best=True,
        )

        # Log final metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metric("final_train_loss", history["train_losses"][-1])
                mlflow.log_metric("final_val_loss", history["val_losses"][-1])
                mlflow.log_metric("best_val_loss", trainer.best_val_loss)
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        # Save training history
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        history_path = results_dir / "training_history.json"
        import json
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Training history saved to {history_path}")
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
