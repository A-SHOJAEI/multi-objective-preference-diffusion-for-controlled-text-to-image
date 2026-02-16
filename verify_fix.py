#!/usr/bin/env python3
"""Verification script to check if the channel mismatch error is fixed."""

import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

print("=" * 60)
print("VERIFICATION: Channel Mismatch Fix")
print("=" * 60)

# Test 1: Import model
print("\n[1/5] Testing model import...")
try:
    from multi_objective_preference_diffusion_for_controlled_text_to_image.models import PreferenceDiffusionModel
    print("✓ Model import successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

# Test 2: Create model instance
print("\n[2/5] Testing model instantiation...")
try:
    model = PreferenceDiffusionModel(
        num_objectives=3,
        use_dynamic_guidance=True,
        guidance_scale=7.5,
        num_inference_steps=50,
    )
    print("✓ Model instantiation successful")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 3: Test forward pass with 3-channel RGB images
print("\n[3/5] Testing forward pass with RGB images...")
try:
    batch_size = 2
    # Create 3-channel RGB images (as would come from data loader)
    images = torch.rand(batch_size, 3, 32, 32)
    text_embeddings = torch.rand(batch_size, 77, 512)
    preference_targets = torch.rand(batch_size, 3) * 0.5 + 0.5

    print(f"   Input image shape: {images.shape} (expecting [B, 3, H, W])")

    outputs = model(
        images,
        text_embeddings,
        preference_targets,
        return_dict=True,
    )

    print(f"   Output loss: {outputs['loss'].item():.4f}")
    print(f"   Diffusion loss: {outputs['diffusion_loss'].item():.4f}")
    print(f"   Preference loss: {outputs['preference_loss'].item():.4f}")
    print("✓ Forward pass successful - channel mismatch FIXED!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test image encoding/decoding
print("\n[4/5] Testing image encoding/decoding...")
try:
    test_images = torch.rand(2, 3, 256, 256)
    print(f"   Input: {test_images.shape} (3 channels RGB)")

    # Test encoding
    latents = model._encode_images(test_images)
    print(f"   Encoded latents: {latents.shape} (should be 4 channels)")

    assert latents.shape[1] == 4, f"Expected 4 channels, got {latents.shape[1]}"

    # Test decoding
    decoded = model._decode_latents(latents)
    print(f"   Decoded images: {decoded.shape} (should be 3 channels RGB)")

    assert decoded.shape[1] == 3, f"Expected 3 channels, got {decoded.shape[1]}"

    print("✓ Encoding/decoding successful")
except Exception as e:
    print(f"✗ Encoding/decoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test generation
print("\n[5/5] Testing image generation...")
try:
    model.eval()
    with torch.no_grad():
        prompts = ["a photo of a cat"]
        preference_targets = torch.tensor([[0.8, 0.7, 0.9]])

        generated = model.generate(
            prompts=prompts,
            preference_targets=preference_targets,
            num_inference_steps=5,  # Small number for quick test
        )

        print(f"   Generated image shape: {generated.shape}")
        assert generated.shape[1] == 3, f"Expected 3 channels, got {generated.shape[1]}"
        print("✓ Image generation successful")
except Exception as e:
    print(f"✗ Image generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL VERIFICATIONS PASSED! ✓")
print("=" * 60)
print("\nThe channel mismatch error has been successfully fixed.")
print("The model now correctly:")
print("  • Accepts 3-channel RGB images as input")
print("  • Converts them to 4-channel latents internally")
print("  • Processes with the UNet expecting 4 channels")
print("  • Converts back to 3-channel RGB for output")
print("\nYou can now run: python scripts/train.py")
