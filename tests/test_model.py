"""Tests for model components."""

import pytest
import torch

from multi_objective_preference_diffusion_for_controlled_text_to_image.models import (
    PreferenceDiffusionModel,
    ParetoWeightedLoss,
    DynamicGuidanceScheduler,
    PreferenceRewardModel,
)


class TestParetoWeightedLoss:
    """Tests for ParetoWeightedLoss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = ParetoWeightedLoss(num_objectives=3)

        predictions = torch.rand(8, 3)
        targets = torch.rand(8, 3)

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0

    def test_return_weights(self):
        """Test returning weights."""
        loss_fn = ParetoWeightedLoss(num_objectives=3)

        predictions = torch.rand(8, 3)
        targets = torch.rand(8, 3)

        loss, weights = loss_fn(predictions, targets, return_weights=True)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (3,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)

    def test_history_tracking(self):
        """Test objective history tracking."""
        loss_fn = ParetoWeightedLoss(num_objectives=3)

        for _ in range(10):
            predictions = torch.rand(4, 3)
            targets = torch.rand(4, 3)
            loss = loss_fn(predictions, targets)

        assert loss_fn.history_idx.item() == 10


class TestDynamicGuidanceScheduler:
    """Tests for DynamicGuidanceScheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = DynamicGuidanceScheduler(
            base_guidance=7.5,
            min_guidance=3.0,
            max_guidance=15.0,
        )

        assert scheduler.current_guidance == 7.5

    def test_guidance_adaptation(self):
        """Test guidance scale adaptation."""
        scheduler = DynamicGuidanceScheduler(base_guidance=7.5)

        # High uncertainty should reduce guidance
        guidance_high_uncertainty = scheduler.step(preference_uncertainty=0.8)

        scheduler.reset()

        # Low uncertainty should keep guidance higher
        guidance_low_uncertainty = scheduler.step(preference_uncertainty=0.2)

        assert guidance_high_uncertainty < guidance_low_uncertainty

    def test_guidance_bounds(self):
        """Test guidance stays within bounds."""
        scheduler = DynamicGuidanceScheduler(
            base_guidance=7.5,
            min_guidance=3.0,
            max_guidance=15.0,
        )

        for _ in range(50):
            uncertainty = torch.rand(1).item()
            guidance = scheduler.step(uncertainty)

            assert 3.0 <= guidance <= 15.0

    def test_reset(self):
        """Test scheduler reset."""
        scheduler = DynamicGuidanceScheduler(base_guidance=7.5)

        for _ in range(10):
            scheduler.step(0.5)

        scheduler.reset()

        assert scheduler.current_guidance == 7.5
        assert len(scheduler.uncertainty_history) == 0


class TestPreferenceRewardModel:
    """Tests for PreferenceRewardModel."""

    def test_model_creation(self):
        """Test model can be created."""
        model = PreferenceRewardModel(
            feature_dim=768,
            hidden_dim=512,
            num_objectives=3,
        )

        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = PreferenceRewardModel(
            feature_dim=768,
            num_objectives=3,
        )

        features = torch.rand(8, 768)
        predictions, uncertainties = model(features)

        assert predictions.shape == (8, 3)
        assert uncertainties.shape == (8, 3)

        # Check predictions in valid range
        assert predictions.min() >= 0
        assert predictions.max() <= 1

        # Check uncertainties are positive
        assert uncertainties.min() >= 0

    def test_compute_reward(self):
        """Test reward computation."""
        model = PreferenceRewardModel(num_objectives=3)

        predictions = torch.rand(8, 3)
        targets = torch.rand(8, 3)

        rewards = model.compute_reward(predictions, targets)

        assert rewards.shape == (8,)


class TestPreferenceDiffusionModel:
    """Tests for PreferenceDiffusionModel."""

    def test_model_creation(self):
        """Test model creation."""
        model = PreferenceDiffusionModel(
            num_objectives=3,
            use_dynamic_guidance=True,
        )

        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        model = PreferenceDiffusionModel(num_objectives=3)

        batch_size = 4
        images = torch.rand(batch_size, 3, 256, 256)  # RGB images, not latents
        text_embeddings = torch.rand(batch_size, 77, 512)
        preference_targets = torch.rand(batch_size, 3)

        outputs = model(
            images,
            text_embeddings,
            preference_targets,
            return_dict=True,
        )

        assert "loss" in outputs
        assert "diffusion_loss" in outputs
        assert "preference_loss" in outputs
        assert outputs["loss"].item() >= 0

    def test_generation(self):
        """Test image generation."""
        model = PreferenceDiffusionModel(
            num_objectives=3,
            num_inference_steps=10,
        )
        model.eval()

        prompts = ["a photo of a cat", "a painting of a sunset"]
        preference_targets = torch.tensor([[0.8, 0.7, 0.9], [0.6, 0.8, 0.7]])

        with torch.no_grad():
            generated = model.generate(
                prompts=prompts,
                preference_targets=preference_targets,
                num_inference_steps=10,
            )

        assert generated.shape[0] == 2
        assert generated.shape[1] == 3  # RGB channels (decoded from latents)
        assert generated.ndim == 4

    def test_dynamic_guidance(self):
        """Test dynamic guidance is used."""
        model = PreferenceDiffusionModel(
            num_objectives=3,
            use_dynamic_guidance=True,
        )

        assert model.use_dynamic_guidance
        assert hasattr(model, "guidance_scheduler")
