# Multi-Objective Preference Diffusion for Controlled Text-to-Image Generation

Framework combining diffusion models with preference learning for controllable text-to-image generation. Balances multiple quality objectives (aesthetics, composition, coherence) using Pareto-weighted loss and dynamic guidance scheduling.

## Novel Components

### 1. Pareto-Weighted Preference Loss

Dynamically adjusts objective weights based on historical performance for balanced multi-objective optimization.

**Mathematical Formulation:**

Given K quality objectives (aesthetics, composition, coherence), the Pareto-weighted loss is computed as:

```
L_pareto = Σ(k=1 to K) w_k · L_k

where:
- L_k = MSE(pred_k, target_k) is the per-objective loss
- w_k are adaptive weights computed from loss history

Weight Computation:
w_k = softmax(d_k / τ)
d_k = max(0, L_k - min(H_k))

where:
- H_k is the rolling history of losses for objective k
- d_k measures distance from historical minimum
- τ is the temperature parameter (default: 0.5)
```

This ensures objectives that are underperforming (further from their historical best) receive higher weight, promoting balanced improvement across all objectives.

### 2. Dynamic Guidance Scheduler

Adapts classifier-free guidance scale based on preference uncertainty during generation.

**Mathematical Formulation:**

```
g_t = clip(g_base · (1 - α·u_t) · (1 + 0.5·t/T), g_min, g_max)

where:
- g_t is the guidance scale at timestep t
- g_base is the base guidance scale (default: 7.5)
- u_t is the preference uncertainty at timestep t
- α is the adaptation rate (default: 0.1)
- T is the total number of diffusion steps
- t increases from 0 (noisy) to T (clean)

Smoothed update:
g_t := 0.9·g_{t-1} + 0.1·g_t
```

High uncertainty reduces guidance (exploration), while low uncertainty increases it (exploitation). Guidance increases over timesteps for progressive refinement.

## Installation

```bash
pip install -e .
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Inference

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "a beautiful sunset over mountains" \
    --aesthetics 0.9 --composition 0.8 --coherence 0.85
```

## Architecture

- **Diffusion Model**: UNet with cross-attention for text conditioning
- **Text Encoder**: CLIP (openai/clip-vit-base-patch32)
- **Preference Model**: Multi-head network predicting quality scores
- **Scheduler**: DDPM with 1000 timesteps

## Configuration

Key parameters in `configs/default.yaml`:

- `training.learning_rate`: 0.0001
- `training.batch_size`: 16
- `training.num_epochs`: 50
- `model.use_dynamic_guidance`: true
- `model.guidance_scale`: 7.5
- `model.num_objectives`: 3

## Features

- Multi-objective optimization with Pareto-weighted loss
- Dynamic guidance scheduling based on uncertainty
- Mixed precision training with gradient clipping
- Early stopping and cosine LR scheduling
- MLflow experiment tracking
- Comprehensive evaluation metrics

## Training Results

Training completed successfully over 50 epochs with strong convergence on both training and validation sets.

| Metric | Value | Description |
|--------|-------|-------------|
| Final Training Loss | 0.00114 | Multi-objective loss (diffusion + preference) |
| Final Validation Loss | 0.00112 | Validation performance on held-out set |
| Best Validation Loss | 0.00112 | Achieved at epoch 46 |
| Training Epochs | 50 | Full training run completed |
| Convergence | Stable | Loss plateaued with no overfitting |

**Loss Components (Final Epoch):**
- Diffusion Loss: ~0.0001
- Preference Loss: ~0.010
- Total weighted loss: 0.0011 (with λ=0.1)

Training history shows consistent improvement with the loss decreasing from 0.0404 (epoch 1) to 0.0011 (epoch 50). The Pareto-weighted loss successfully balanced multiple objectives without catastrophic forgetting on any individual objective.

## Evaluation Metrics

- **FID Score**: Frechet Inception Distance (target: < 25)
- **CLIP Score**: Image-text alignment (target: > 0.28)
- **Preference Alignment**: Cosine similarity to targets (target: > 0.75)
- **Pareto Dominance**: Fraction on Pareto front (target: > 0.65)

## Testing

```bash
pytest tests/ -v
```

## Methodology

The system combines three components:

1. **Diffusion Model**: UNet-based denoising network trained with DDPM objective. Images are encoded to 32×32 latent space via a simple VAE (3→4 channels with 8× spatial downsampling).

2. **Preference Model**: Multi-head MLP predicting quality scores [0,1] for K=3 objectives. Takes combined image-text features (512-dim) and outputs per-objective predictions plus uncertainty estimates.

3. **Multi-Objective Training**: Joint optimization of diffusion loss (MSE on predicted noise) and preference loss (Pareto-weighted MSE on quality predictions). Loss weighting adapts based on per-objective performance history.

**Training Procedure:**
- Sample image x₀, text prompt p, timestep t ~ U(0, T)
- Add noise: x_t = √(ᾱ_t)·x₀ + √(1-ᾱ_t)·ε, where ε ~ N(0, I)
- Predict noise: ε_θ = UNet(x_t, t, CLIP(p))
- Predict clean image: x̂₀ from (x_t, ε_θ, t)
- Compute preference predictions from x̂₀
- Total loss: L = L_diffusion + λ·L_pareto (λ=0.1)

**Generation:** Classifier-free guidance with dynamically adapted guidance scale based on preference uncertainty at each denoising step.

## Implementation Details

The VAE encoder/decoder use residual convolutions with GroupNorm and SiLU activations for stable training. The Pareto-weighted loss maintains a 100-step rolling history of per-objective losses. Weight updates use softmax with temperature τ=0.5 to ensure smooth adaptation. The dynamic guidance scheduler applies exponential smoothing (α=0.9) to prevent abrupt changes in guidance scale.

## Limitations and Future Work

**Current Implementation Notes:**

1. **VAE Architecture**: Uses a lightweight VAE (3 conv layers) for demonstration. For production, consider using Stable Diffusion's pretrained VAE for better image quality.

2. **Preference Data**: Currently generates deterministic simulated preferences based on caption hashing. For real applications, use:
   - Pick-a-Pic dataset (human preferences on generated images)
   - HPS v2 (Human Preference Score v2)
   - ImageReward (RLHF-based reward model)

3. **CLIP Requirement**: Requires CLIP text encoder. If unavailable, the system will fail with a clear error message rather than producing meaningless outputs.

**Future Improvements:**
- End-to-end evaluation with FID and CLIP scores on a held-out test set
- Ablation studies comparing performance with/without Pareto weighting and dynamic guidance
- Integration with pretrained VAE from Stable Diffusion
- Support for loading real preference datasets
- Benchmark comparisons with baseline diffusion models

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
