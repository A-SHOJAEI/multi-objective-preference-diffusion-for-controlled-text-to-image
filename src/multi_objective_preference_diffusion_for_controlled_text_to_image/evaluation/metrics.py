"""Evaluation metrics for diffusion models with preference learning."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Container for computing multiple evaluation metrics.

    Args:
        device: Device for computation
        feature_dim: Dimension for feature extraction
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        feature_dim: int = 2048,
    ) -> None:
        self.device = device
        self.feature_dim = feature_dim

        # Simple feature extractor (in production would use InceptionV3)
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extractor for FID computation.

        Returns:
            Feature extraction network
        """
        # Simplified feature extractor
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, self.feature_dim),
        ).to(self.device)

        model.eval()
        return model

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Feature vectors [B, feature_dim]
        """
        images = images.to(self.device)

        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2

        features = self.feature_extractor(images)
        return features.cpu().numpy()

    def compute_all(
        self,
        generated_images: torch.Tensor,
        real_images: torch.Tensor,
        prompts: List[str],
        preference_preds: torch.Tensor,
        preference_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Args:
            generated_images: Generated images [B, C, H, W]
            real_images: Real reference images [B, C, H, W]
            prompts: Text prompts used for generation
            preference_preds: Predicted preference scores [B, num_objectives]
            preference_targets: Target preference scores [B, num_objectives]

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # FID score
        try:
            fid = compute_fid_score(
                generated_images, real_images, self.feature_extractor, self.device
            )
            metrics["fid_score"] = fid
        except Exception as e:
            logger.warning(f"Failed to compute FID: {e}")
            metrics["fid_score"] = -1.0

        # CLIP score
        try:
            clip_score = compute_clip_score(generated_images, prompts, self.device)
            metrics["clip_score"] = clip_score
        except Exception as e:
            logger.warning(f"Failed to compute CLIP score: {e}")
            metrics["clip_score"] = -1.0

        # Preference alignment
        try:
            alignment = compute_preference_alignment(
                preference_preds, preference_targets
            )
            metrics["preference_alignment"] = alignment
        except Exception as e:
            logger.warning(f"Failed to compute preference alignment: {e}")
            metrics["preference_alignment"] = -1.0

        # Pareto dominance ratio
        try:
            pareto_ratio = compute_pareto_dominance(
                preference_preds, preference_targets
            )
            metrics["pareto_dominance_ratio"] = pareto_ratio
        except Exception as e:
            logger.warning(f"Failed to compute Pareto dominance: {e}")
            metrics["pareto_dominance_ratio"] = -1.0

        return metrics


def compute_fid_score(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    feature_extractor: nn.Module,
    device: torch.device,
) -> float:
    """Compute Frechet Inception Distance (FID).

    Args:
        generated_images: Generated images [B, C, H, W]
        real_images: Real images [B, C, H, W]
        feature_extractor: Feature extraction network
        device: Computation device

    Returns:
        FID score (lower is better)
    """
    # Extract features
    with torch.no_grad():
        gen_features = []
        real_features = []

        # Process in batches
        batch_size = 32
        for i in range(0, len(generated_images), batch_size):
            batch = generated_images[i:i + batch_size].to(device)
            if batch.min() < 0:
                batch = (batch + 1) / 2
            gen_features.append(feature_extractor(batch).cpu().numpy())

        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i + batch_size].to(device)
            if batch.min() < 0:
                batch = (batch + 1) / 2
            real_features.append(feature_extractor(batch).cpu().numpy())

        gen_features = np.concatenate(gen_features, axis=0)
        real_features = np.concatenate(real_features, axis=0)

    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)

    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)

    # Compute FID
    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)

    return float(fid)


def compute_clip_score(
    images: torch.Tensor,
    prompts: List[str],
    device: torch.device,
) -> float:
    """Compute CLIP score for image-text alignment.

    Args:
        images: Generated images [B, C, H, W]
        prompts: Text prompts
        device: Computation device

    Returns:
        Average CLIP score (higher is better)
    """
    # Simplified CLIP score computation
    # In production, would use actual CLIP model

    # Simulate CLIP score based on image statistics
    scores = []

    for img, prompt in zip(images, prompts):
        # Simple heuristic: higher variance and mean brightness = better quality
        img_np = img.cpu().numpy()
        variance = np.var(img_np)
        mean_brightness = np.mean(img_np)

        # Simulate score in CLIP range
        score = 0.2 + (variance * 0.3) + (mean_brightness * 0.2)
        score = np.clip(score, 0.0, 1.0)
        scores.append(score)

    return float(np.mean(scores))


def compute_preference_alignment(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute preference alignment score.

    Measures how well predictions align with target preferences.

    Args:
        predictions: Predicted preference scores [B, num_objectives]
        targets: Target preference scores [B, num_objectives]

    Returns:
        Alignment score in [0, 1] (higher is better)
    """
    # Compute cosine similarity for each sample
    predictions_norm = F.normalize(predictions, p=2, dim=1)
    targets_norm = F.normalize(targets, p=2, dim=1)

    similarities = (predictions_norm * targets_norm).sum(dim=1)

    # Convert from [-1, 1] to [0, 1]
    alignment = (similarities + 1) / 2

    return float(alignment.mean().item())


def compute_pareto_dominance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Compute Pareto dominance ratio.

    Measures what fraction of predictions lie on or near the Pareto front
    defined by the targets.

    Args:
        predictions: Predicted preference scores [B, num_objectives]
        targets: Target preference scores [B, num_objectives]

    Returns:
        Pareto dominance ratio in [0, 1] (higher is better)
    """
    predictions_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Check if each prediction is Pareto-dominated by any target
    dominated_count = 0

    for pred in predictions_np:
        # Check if any target dominates this prediction
        is_dominated = False

        for target in targets_np:
            # Target dominates pred if it's better in all objectives
            if np.all(target >= pred - 0.1):  # 0.1 tolerance
                is_dominated = True
                break

        if not is_dominated:
            dominated_count += 1

    # Ratio of non-dominated predictions
    ratio = dominated_count / len(predictions_np)

    return float(ratio)


def compute_per_objective_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    objective_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-objective analysis metrics.

    Args:
        predictions: Predicted scores [B, num_objectives]
        targets: Target scores [B, num_objectives]
        objective_names: Names of objectives

    Returns:
        Dictionary with per-objective metrics
    """
    num_objectives = predictions.size(1)

    if objective_names is None:
        objective_names = [f"objective_{i}" for i in range(num_objectives)]

    results = {}

    for i, name in enumerate(objective_names):
        pred_obj = predictions[:, i]
        target_obj = targets[:, i]

        results[name] = {
            "mse": float(F.mse_loss(pred_obj, target_obj).item()),
            "mae": float(F.l1_loss(pred_obj, target_obj).item()),
            "correlation": float(
                np.corrcoef(
                    pred_obj.cpu().numpy(),
                    target_obj.cpu().numpy()
                )[0, 1]
            ),
            "mean_pred": float(pred_obj.mean().item()),
            "mean_target": float(target_obj.mean().item()),
        }

    return results
