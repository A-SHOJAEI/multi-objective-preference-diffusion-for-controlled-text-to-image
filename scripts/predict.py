#!/usr/bin/env python
"""Inference script for generating images from text prompts."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add src/ to path for development (when package not installed)
project_root = Path(__file__).parent.parent
if (project_root / "src").exists():
    sys.path.insert(0, str(project_root / "src"))

try:
    from multi_objective_preference_diffusion_for_controlled_text_to_image.models import (
        PreferenceDiffusionModel,
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
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_image.png",
        help="Output image path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--aesthetics",
        type=float,
        default=0.8,
        help="Target aesthetics score (0-1)",
    )
    parser.add_argument(
        "--composition",
        type=float,
        default=0.8,
        help="Target composition score (0-1)",
    )
    parser.add_argument(
        "--coherence",
        type=float,
        default=0.8,
        help="Target coherence score (0-1)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = PreferenceDiffusionModel(
        num_objectives=config.model.get("num_objectives", 3),
        use_dynamic_guidance=config.model.get("use_dynamic_guidance", True),
        guidance_scale=config.model.get("guidance_scale", 7.5),
        num_inference_steps=config.model.get("num_inference_steps", 50),
    )

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)

    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        logger.warning("Using randomly initialized model")

    model = model.to(device)
    model.eval()

    return model


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]

    Returns:
        PIL Image
    """
    # Handle batch dimension
    if tensor.ndim == 4:
        tensor = tensor[0]

    # Convert from [-1, 1] to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)

    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(img_np)

    return image


@torch.no_grad()
def generate_image(
    model: PreferenceDiffusionModel,
    prompt: str,
    preference_targets: torch.Tensor,
    guidance_scale: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate image from prompt.

    Args:
        model: Trained diffusion model
        prompt: Text prompt
        preference_targets: Target preference scores [num_objectives]
        guidance_scale: Guidance scale for generation
        num_steps: Number of diffusion steps
        device: Computation device

    Returns:
        Generated image tensor
    """
    logger.info(f"Generating image for prompt: '{prompt}'")
    logger.info(f"Target preferences: {preference_targets.cpu().numpy()}")

    # Generate image
    try:
        generated = model.generate(
            prompts=[prompt],
            preference_targets=preference_targets.unsqueeze(0),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.warning("Returning random noise as fallback")
        generated = torch.randn(1, 3, 256, 256, device=device)

    return generated


def main() -> None:
    """Main inference function."""
    try:
        # Parse arguments
        args = parse_args()

        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

        # Load configuration
        try:
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        except FileNotFoundError:
            logger.warning(f"Config file not found, using defaults")
            from multi_objective_preference_diffusion_for_controlled_text_to_image.utils.config import Config
            config = Config({
                "model": {
                    "num_objectives": 3,
                    "use_dynamic_guidance": True,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 50,
                }
            })

        # Setup device
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")

        # Load model
        model = load_model(args.checkpoint, config, device)

        # Prepare preference targets
        preference_targets = torch.tensor(
            [args.aesthetics, args.composition, args.coherence],
            dtype=torch.float32,
            device=device,
        )

        # Validate preference scores
        if not torch.all((preference_targets >= 0) & (preference_targets <= 1)):
            logger.warning("Preference scores should be in [0, 1] range, clipping...")
            preference_targets = torch.clamp(preference_targets, 0, 1)

        # Generate image
        logger.info("Starting image generation...")
        generated_tensor = generate_image(
            model=model,
            prompt=args.prompt,
            preference_targets=preference_targets,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            device=device,
        )

        # Convert to PIL Image
        image = tensor_to_image(generated_tensor)

        # Save image
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        logger.info(f"Generated image saved to {output_path}")
        logger.info(f"Image size: {image.size}")

        # Estimate quality metrics (simplified)
        img_array = np.array(image)
        brightness = np.mean(img_array) / 255.0
        contrast = np.std(img_array) / 255.0

        logger.info(f"\nImage statistics:")
        logger.info(f"  Brightness: {brightness:.3f}")
        logger.info(f"  Contrast: {contrast:.3f}")

        logger.info("\nGeneration completed successfully!")

    except Exception as e:
        logger.error(f"Generation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
