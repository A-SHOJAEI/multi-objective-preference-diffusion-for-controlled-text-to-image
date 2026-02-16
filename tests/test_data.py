"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from multi_objective_preference_diffusion_for_controlled_text_to_image.data import (
    ConceptualCaptionsDataset,
    PreferenceDataset,
    ImagePreprocessor,
    TextPreprocessor,
)


class TestConceptualCaptionsDataset:
    """Tests for ConceptualCaptionsDataset."""

    def test_dataset_creation(self):
        """Test dataset can be created."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=100,
            image_size=256,
        )
        assert len(dataset) > 0

    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=10,
            image_size=256,
        )

        item = dataset[0]
        assert "image" in item
        assert "caption" in item
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 256, 256)

    def test_dataset_length(self):
        """Test dataset length."""
        max_samples = 50
        dataset = ConceptualCaptionsDataset(
            split="train",
            max_samples=max_samples,
        )
        assert len(dataset) <= max_samples


class TestPreferenceDataset:
    """Tests for PreferenceDataset."""

    def test_preference_dataset_creation(self):
        """Test preference dataset creation."""
        dataset = PreferenceDataset(
            split="train",
            max_samples=100,
        )
        assert len(dataset) > 0

    def test_preference_dataset_getitem(self):
        """Test getting items from preference dataset."""
        dataset = PreferenceDataset(
            split="train",
            max_samples=10,
        )

        item = dataset[0]
        assert "prompt" in item
        assert "aesthetics" in item
        assert "composition" in item
        assert "coherence" in item

        # Check scores are in valid range
        assert 0 <= item["aesthetics"].item() <= 1
        assert 0 <= item["composition"].item() <= 1
        assert 0 <= item["coherence"].item() <= 1


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    def test_preprocessor_normalization(self):
        """Test image normalization."""
        preprocessor = ImagePreprocessor(
            image_size=256,
            normalize=True,
            augment=False,
        )

        image = torch.rand(3, 256, 256)
        processed = preprocessor(image)

        # Check normalized to [-1, 1]
        assert processed.min() >= -1.0
        assert processed.max() <= 1.0

    def test_preprocessor_denormalization(self):
        """Test image denormalization."""
        preprocessor = ImagePreprocessor(normalize=True)

        image = torch.rand(3, 256, 256)
        normalized = preprocessor(image)
        denormalized = preprocessor.denormalize(normalized)

        # Should be close to original range [0, 1]
        assert denormalized.min() >= 0.0
        assert denormalized.max() <= 1.0

    def test_preprocessor_resize(self):
        """Test image resizing."""
        preprocessor = ImagePreprocessor(image_size=128, normalize=False)

        image = torch.rand(3, 256, 256)
        processed = preprocessor(image)

        assert processed.shape == (3, 128, 128)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_text_encoding(self):
        """Test text encoding."""
        preprocessor = TextPreprocessor(max_length=77)

        text = "A photo of a cat"
        encoded = preprocessor(text, return_tensors=True)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape[0] == 77

    def test_text_decoding(self):
        """Test text decoding."""
        preprocessor = TextPreprocessor(max_length=77)

        text = "A photo of a cat"
        encoded = preprocessor(text, return_tensors=True)
        decoded = preprocessor.decode(encoded)

        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_truncation(self):
        """Test text truncation."""
        preprocessor = TextPreprocessor(max_length=10, truncation=True)

        long_text = "This is a very long text that should be truncated"
        encoded = preprocessor(long_text, return_tensors=True)

        assert encoded.shape[0] == 10
