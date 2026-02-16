"""Training loop with learning rate scheduling and early stopping."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PreferenceDiffusionTrainer:
    """Trainer for preference-guided diffusion models.

    Args:
        model: The diffusion model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Maximum gradient norm for clipping
        checkpoint_dir: Directory to save checkpoints
        log_interval: Steps between logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # History tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Mixed precision training: {self.use_amp}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        diffusion_loss_sum = 0.0
        preference_loss_sum = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch["image"].to(self.device)
            captions = batch["caption"]

            # Get text embeddings using model's CLIP encoder
            text_embeddings = self.model.encode_prompts(captions, device=self.device)

            # Get preference targets from batch if available, otherwise use defaults
            if "preference_targets" in batch:
                preference_targets = batch["preference_targets"].to(self.device)
            else:
                # Default: balanced objectives at 0.75
                batch_size = images.size(0)
                preference_targets = torch.full(
                    (batch_size, 3), 0.75, device=self.device, dtype=torch.float32
                )

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images, text_embeddings, preference_targets, return_dict=True
                    )
                    loss = outputs["loss"]

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    images, text_embeddings, preference_targets, return_dict=True
                )
                loss = outputs["loss"]

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.optimizer.step()

            # Track losses
            epoch_loss += loss.item()
            diffusion_loss_sum += outputs["diffusion_loss"].item()
            preference_loss_sum += outputs["preference_loss"].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "diff": f"{outputs['diffusion_loss'].item():.4f}",
                "pref": f"{outputs['preference_loss'].item():.4f}",
            })

            # Periodic logging
            if (batch_idx + 1) % self.log_interval == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={loss.item():.4f}, "
                    f"diffusion={outputs['diffusion_loss'].item():.4f}, "
                    f"preference={outputs['preference_loss'].item():.4f}"
                )

        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_diffusion = diffusion_loss_sum / num_batches
        avg_preference = preference_loss_sum / num_batches

        self.train_losses.append(avg_loss)

        return {
            "train_loss": avg_loss,
            "train_diffusion_loss": avg_diffusion,
            "train_preference_loss": avg_preference,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        diffusion_loss_sum = 0.0
        preference_loss_sum = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch["image"].to(self.device)
            captions = batch["caption"]
            batch_size = images.size(0)

            # Get text embeddings using model's CLIP encoder
            text_embeddings = self.model.encode_prompts(captions, device=self.device)

            # Get preference targets from batch if available, otherwise use defaults
            if "preference_targets" in batch:
                preference_targets = batch["preference_targets"].to(self.device)
            else:
                # Default: balanced objectives at 0.75
                batch_size = images.size(0)
                preference_targets = torch.full(
                    (batch_size, 3), 0.75, device=self.device, dtype=torch.float32
                )

            # Forward pass
            outputs = self.model(
                images, text_embeddings, preference_targets, return_dict=True
            )

            val_loss += outputs["loss"].item()
            diffusion_loss_sum += outputs["diffusion_loss"].item()
            preference_loss_sum += outputs["preference_loss"].item()
            num_batches += 1

        # Compute metrics
        avg_val_loss = val_loss / num_batches
        avg_diffusion = diffusion_loss_sum / num_batches
        avg_preference = preference_loss_sum / num_batches

        self.val_losses.append(avg_val_loss)

        return {
            "val_loss": avg_val_loss,
            "val_diffusion_loss": avg_diffusion,
            "val_preference_loss": avg_preference,
        }

    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
        save_best: bool = True,
    ) -> Dict[str, List[float]]:
        """Main training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model checkpoint

        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{num_epochs}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"lr={current_lr:.6f}"
            )

            # Save best model
            if save_best and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")
                self.patience_counter = 0
                logger.info(f"New best model saved with val_loss={self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={early_stopping_patience})"
                )
                break

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint("final_model.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
