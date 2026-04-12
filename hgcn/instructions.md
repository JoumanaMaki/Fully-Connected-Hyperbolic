# HGCN Experiment Instructions

## Overview

These experiments integrate our `LorentzFullyConnectedOurs` layer (signed geodesic distance to learned hyperplanes) into the HGCN (Hyperbolic Graph Convolutional Network) framework from Chami et al. 2019. The goal is to demonstrate that our geometrically-principled hyperbolic linear layer improves performance on tasks with naturally hierarchical structure.

## What was changed

Four files were modified from the original HGCN codebase:

- **`layers/hyp_layers.py`** — Added `HypLinearOurs`, a drop-in replacement for `HypLinear`. Instead of logmap → Euclidean linear → expmap, it constructs spacelike normal vectors (parameterized by direction `U` and offset `a`) and computes Minkowski inner products with input points, then projects back to the hyperboloid. Uses weight normalization (direction `v` + scale `g`).

- **`models/decoders.py`** — Added `LorentzMLRDecoder`, which classifies by computing signed geodesic distances to per-class hyperplanes: `(1/sqrt(c)) * asinh(sqrt(c) * <x, V>_L)`. This replaces the standard logmap → linear → softmax decoder.

- **`models/encoders.py`** — Passes `--linear-variant` through the HGCN encoder to `HyperbolicGraphConvolution`.

- **`models/base_models.py`** — Routes `--decoder-variant` to select between `LinearDecoder` and `LorentzMLRDecoder`.

- **`config.py`** — Added two flags: `--linear-variant` and `--decoder-variant`.

## New flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--linear-variant` | `standard`, `ours` | `standard` | Encoder linear layer variant |
| `--decoder-variant` | `standard`, `mlr` | `standard` | Decoder variant (NC tasks only) |

## Setup

```bash
# From the hgcn/ directory
export DATAPATH=./data
export PYTHONPATH=.
```

The datasets (Disease, Airport, Cora, Pubmed) are already in `data/`. The code runs with the project's existing `.venv` (Python 3.13, PyTorch 2.9).

## Running experiments

### Full sweep (all datasets, 5 seeds)

```bash
bash run_experiments.sh
```

This runs 5 seeds across Disease NC/LP, Airport NC, and Cora NC/LP, comparing:
1. **Standard** — original HGCN (`--linear-variant standard --decoder-variant standard`)
2. **Ours (encoder only)** — our linear layer + standard decoder
3. **Ours (encoder + MLR)** — our linear layer + MLR decoder

### Individual experiments

```bash
# Disease NC — most tree-like (delta ~ 0), strongest case for hyperbolic
python train.py --task nc --dataset disease_nc --model HGCN --dim 16 \
    --num-layers 2 --manifold Hyperboloid --c None --lr 0.01 --dropout 0 \
    --linear-variant ours --decoder-variant mlr --grad-clip 1

# Disease LP
python train.py --task lp --dataset disease_lp --model HGCN --dim 16 \
    --num-layers 2 --manifold Hyperboloid --c None --lr 0.01 --dropout 0 \
    --linear-variant ours --grad-clip 1

# Airport NC — hierarchical route network
python train.py --task nc --dataset airport --model HGCN --dim 16 \
    --num-layers 2 --manifold Hyperboloid --c None --lr 0.01 --dropout 0 \
    --linear-variant ours --decoder-variant mlr --grad-clip 1

# Cora NC — citation graph (less hyperbolic, sanity check)
python train.py --task nc --dataset cora --model HGCN --dim 16 \
    --num-layers 2 --manifold Hyperboloid --c None --lr 0.01 --dropout 0.5 \
    --weight-decay 0.001 --linear-variant ours --decoder-variant mlr --grad-clip 1

# Cora LP
python train.py --task lp --dataset cora --model HGCN --dim 16 \
    --num-layers 2 --manifold Hyperboloid --c None --lr 0.01 --dropout 0.5 \
    --weight-decay 0.001 --linear-variant ours --grad-clip 1
```

## Datasets and expected behavior

| Dataset | Task | Gromov delta | Structure | Expected outcome |
|---------|------|-------------|-----------|-----------------|
| Disease | NC, LP | ~0 | Binary tree (disease ontology) | Largest improvement — maximally hyperbolic |
| Airport | NC | Low | Hierarchical route network | Clear improvement |
| Cora | NC, LP | Higher | Citation graph | Comparable or modest improvement |

The benefit of our layer should correlate with how tree-like the graph is (lower delta = more hyperbolic = larger expected gain).

## Metrics

- **NC tasks**: accuracy, F1 (reported as test set results at best validation F1)
- **LP tasks**: ROC-AUC, AP (reported at best validation)

## Comparison points for the paper

For each dataset, report a table like:

| Method | Encoder | Decoder | Accuracy | F1 |
|--------|---------|---------|----------|----|
| HGCN (Chami et al.) | standard | standard | ... | ... |
| HGCN + ours | ours | standard | ... | ... |
| HGCN + ours + MLR | ours | mlr | ... | ... |

This isolates the effect of (a) our encoder layer and (b) our encoder + decoder together.
