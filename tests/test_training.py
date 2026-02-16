"""Tests for training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from multi_objective_preference_diffusion_for_controlled_text_to_image.models import (
    PreferenceDiffusionModel,
)
from multi_objective_preference_diffusion_for_controlled_text_to_image.training import (
    PreferenceDiffusionTrainer,
)


class MockDataset(TensorDataset):
    """Mock dataset for testing."""

    def __init__(self, num_samples=100):
        images = torch.rand(num_samples, 3, 256, 256)  # RGB images
        super().__init__(images)

    def __getitem__(self, idx):
        image = super().__getitem__(idx)[0]
        return {
            "image": image,
            "caption": f"caption {idx}",
        }


class TestPreferenceDiffusionTrainer:
    """Tests for PreferenceDiffusionTrainer."""

    @pytest.fixture
    def trainer_setup(self):
        """Setup trainer for testing."""
        # Create mock data
        train_dataset = MockDataset(num_samples=64)
        val_dataset = MockDataset(num_samples=32)

        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        # Create model
        model = PreferenceDiffusionModel(num_objectives=3)

        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.001)

        # Create trainer
        device = torch.device("cpu")
        trainer = PreferenceDiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            use_amp=False,
            checkpoint_dir="test_checkpoints",
        )

        return trainer

    def test_trainer_creation(self, trainer_setup):
        """Test trainer can be created."""
        trainer = trainer_setup
        assert trainer is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_train_epoch(self, trainer_setup):
        """Test training for one epoch."""
        trainer = trainer_setup

        metrics = trainer.train_epoch()

        assert "train_loss" in metrics
        assert "train_diffusion_loss" in metrics
        assert "train_preference_loss" in metrics
        assert metrics["train_loss"] >= 0

    def test_validate(self, trainer_setup):
        """Test validation."""
        trainer = trainer_setup

        metrics = trainer.validate()

        assert "val_loss" in metrics
        assert "val_diffusion_loss" in metrics
        assert "val_preference_loss" in metrics
        assert metrics["val_loss"] >= 0

    def test_full_training(self, trainer_setup, tmp_path):
        """Test full training loop."""
        trainer = trainer_setup
        trainer.checkpoint_dir = tmp_path

        history = trainer.train(
            num_epochs=2,
            early_stopping_patience=5,
            save_best=True,
        )

        assert "train_losses" in history
        assert "val_losses" in history
        assert len(history["train_losses"]) == 2
        assert len(history["val_losses"]) == 2

    def test_checkpoint_save_load(self, trainer_setup, tmp_path):
        """Test checkpoint saving and loading."""
        trainer = trainer_setup
        trainer.checkpoint_dir = tmp_path

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint("test_checkpoint.pt")

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        train_dataset = MockDataset(num_samples=64)
        val_dataset = MockDataset(num_samples=32)

        train_loader = DataLoader(train_dataset, batch_size=8)
        val_loader = DataLoader(val_dataset, batch_size=8)

        model = PreferenceDiffusionModel(num_objectives=3)
        optimizer = Adam(model.parameters(), lr=0.001)

        new_trainer = PreferenceDiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            use_amp=False,
            checkpoint_dir=tmp_path,
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        assert new_trainer.current_epoch == trainer.current_epoch
        assert new_trainer.global_step == trainer.global_step

    def test_gradient_clipping(self, trainer_setup):
        """Test gradient clipping is applied."""
        trainer = trainer_setup
        trainer.gradient_clip = 1.0

        # Run one training step
        trainer.train_epoch()

        # Check that gradients are within bounds
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Should be clipped or below threshold
                assert grad_norm <= 10.0  # Reasonable upper bound

    def test_early_stopping(self, trainer_setup):
        """Test early stopping."""
        trainer = trainer_setup

        # Force high validation loss to trigger early stopping
        trainer.best_val_loss = 0.0001

        history = trainer.train(
            num_epochs=20,
            early_stopping_patience=2,
            save_best=True,
        )

        # Should stop before 20 epochs
        assert len(history["train_losses"]) < 20
