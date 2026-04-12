# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase implementing fully connected layers and ResNet-18 on the Lorentz (hyperboloid) model of hyperbolic space. The main contribution is `LorentzFullyConnectedOurs` — a layer that computes signed geodesic distances to learned hyperplanes in Lorentz space, rather than applying Euclidean linear maps and reprojecting (as in Chen et al. 2022).

## Setup

```bash
uv sync          # install all dependencies (Python >= 3.12, PyTorch+CUDA 12.6)
```

The `geoopt` package is installed from source (git). No test suite exists in this repo.

## Running experiments

```bash
# CIFAR-10 training (requires GPU, logs to W&B project "ICML_Hyperbolic")
cd cifar_exp && python main.py

# Toy convergence experiment (compares layer variants at various hyperbolic distances)
cd toy_exp && python toy_exp.py --models ours chen poincare

# Runtime benchmarks (requires GPU)
cd runtime_exp && python my_runtimes.py
```

CIFAR training config is the `default_config` dict in `cifar_exp/main.py`. W&B sweeps override these values.

## Architecture

### Manifold dimension convention

All Lorentz tensors have shape `(..., D)` where `D = spatial_dim + 1`. The first component (index 0) is the **time** coordinate; the rest are **space**. Feature dimensions in layer constructors (e.g. `in_features=65`) include the time component. The hyperboloid constraint is: `-x_0^2 + x_1^2 + ... + x_n^2 = -1/k`.

For image tensors the channel dimension encodes the manifold: shape `[B, C, H, W]` where `C` includes the time component (channel 0). The `manifold_dim=1` argument controls which axis is the manifold dimension for operations like `relu` and `projection_space_orthogonal`.

### Layer data flow (Lorentz ResNet-18)

1. **Input projection** (`LProjection.py`): Euclidean RGB `[B, 3, H, W]` → prepend time via `projection_space_orthogonal` → Lorentz conv + BN → `[B, C, H, W]` on manifold
2. **ResNet stages** (`LResNet.py` → `LResBlock.py`): Each block does conv → BN → ReLU → conv → BN → skip-add → ReLU. Skip connections add in space dimensions only, then recompute time via projection.
3. **Lorentz conv** (`LConv.py`): Unfold patches → `direct_concat` (fuses multiple Lorentz points into one by combining time components and concatenating space) → apply Lorentz FC
4. **Lorentz BN** (`LBatchNorm.py`): Compute Frechet mean centroid → logmap to tangent space → transport to origin → scale by γ/σ → transport to learned center β → expmap back. The `normalisation_mode` flag controls variants (centering-only, fixed gamma, clamped scale, etc.)
5. **Classifier**: Either `LorentzMLR` or `LorentzFullyConnectedOurs` with `do_mlr=True`, producing class logits via signed distance to hyperplanes

### The two FC variants (`LLinear.py`)

- **"ours"** (`LorentzFullyConnectedOurs`): Parameterizes hyperplanes via direction `U` and offset `a`. Constructs spacelike normal vectors, computes `asinh`-based signed distances. Output is projected back to manifold. Supports weight normalization (`use_weight_norm`).
- **"theirs"** (`LorentzFullyConnectedTheirs`): Applies `nn.Linear` directly then reprojects space components onto the hyperboloid. Based on Chen et al. (2022).

Select via `resolve_lorentz_fc_class(variant)` which returns the class. The `fc_variant` config string propagates through ResNet → ResBlock → Conv → FC.

### Numerical stability patterns

- `torch.clamp(x, min=1.0 + eps)` before `acosh` calls (argument must be > 1)
- `torch.clamp(arg, -100, 100)` before `sinh`/`cosh` to prevent overflow
- `torch.where(is_zero, ...)` to handle zero-norm edge cases in exp/log maps
- Projections (`projx`, `projection_space_orthogonal`) recompute time from space to enforce the hyperboloid constraint

### Optimizer setup

CIFAR training uses Riemannian SGD (`geoopt.optim.RiemannianSGD`) with 3 parameter groups: standard params, `ManifoldParameter` params (0.2× LR), and curvature `k` (no weight decay, fixed 1e-4 LR). The first StepLR milestone skips the LR decay for manifold params to let them sync up.

### Baseline implementations

- `poincare.py`: Poincare ball model — linear layer computes distance-to-hyperplanes in the ball, maps back via inverse operations
- `chen.py`: `ChenLinear` — standalone Chen et al. layer (similar to "theirs" but used directly in toy/runtime experiments)
- `bdeir.py`: `BdeirLorentzMLR` — Bdeir et al. classification head
