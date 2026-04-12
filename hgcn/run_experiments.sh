#!/usr/bin/env bash
# Full sweep comparing standard HGCN vs ours vs chen linear layers.
#
# Variants:
#   standard  = original HypLinear (logmap -> Euclidean linear -> expmap), PoincareBall
#   ours      = HypLinearOurs (signed distance to learned hyperplanes), Hyperboloid
#   chen      = HypLinearChen (Chen et al. 2020 Lorentz linear), Hyperboloid
#
# NC tasks also vary decoder-variant (standard for standard, mlr for ours/chen).
# LP tasks omit decoder-variant.
#
# Sweep: 3 LRs × 2 curvature settings × 3 seeds × 3 variants × 4 dataset-tasks = 216 runs

set -euo pipefail

export DATAPATH="${DATAPATH:-./data}"

SEEDS="0 1 2 3 4 5 6"
LRS="0.01 0.05 0.1"
CURVATURES="1.0 None"
COMMON="--num-layers 2 --grad-clip 1 --log-freq 25 --patience 200"

run() {
    local desc="$1"; shift
    echo "========================================"
    echo "  $desc"
    echo "========================================"
    python train.py "$@"
    echo ""
}

# --------------------------------------------------------------------------- #
#  Disease NC
# --------------------------------------------------------------------------- #
for SEED in $SEEDS; do
for LR in $LRS; do
for C in $CURVATURES; do
    run "Disease NC | standard | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset disease_nc --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant standard --decoder-variant standard \
        --seed "$SEED" --manifold PoincareBall --c "$C" $COMMON

    run "Disease NC | ours | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset disease_nc --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant ours --decoder-variant mlr \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON

    run "Disease NC | chen | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset disease_nc --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant chen --decoder-variant mlr \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON
done
done
done

# --------------------------------------------------------------------------- #
#  Disease LP
# --------------------------------------------------------------------------- #
for SEED in $SEEDS; do
for LR in $LRS; do
for C in $CURVATURES; do
    run "Disease LP | standard | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset disease_lp --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant standard --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold PoincareBall --c "$C" $COMMON

    run "Disease LP | ours | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset disease_lp --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant ours --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON

    run "Disease LP | chen | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset disease_lp --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant chen --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON
done
done
done

# --------------------------------------------------------------------------- #
#  Airport NC
# --------------------------------------------------------------------------- #
for SEED in $SEEDS; do
for LR in $LRS; do
for C in $CURVATURES; do
    run "Airport NC | standard | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant standard --decoder-variant standard \
        --seed "$SEED" --manifold PoincareBall --c "$C" $COMMON

    run "Airport NC | ours | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant ours --decoder-variant mlr \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON

    run "Airport NC | chen | lr=$LR | c=$C | seed=$SEED" \
        --task nc --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant chen --decoder-variant mlr \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON
done
done
done

# --------------------------------------------------------------------------- #
#  Airport LP
# --------------------------------------------------------------------------- #
for SEED in $SEEDS; do
for LR in $LRS; do
for C in $CURVATURES; do
    run "Airport LP | standard | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant standard --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold PoincareBall --c "$C" $COMMON

    run "Airport LP | ours | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant ours --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON

    run "Airport LP | chen | lr=$LR | c=$C | seed=$SEED" \
        --task lp --dataset airport --model HGCN --dim 16 --lr "$LR" --dropout 0 \
        --linear-variant chen --normalize-feats 0 --weight-decay 0 \
        --seed "$SEED" --manifold Hyperboloid --c "$C" $COMMON
done
done
done

echo "All experiments complete. (216 runs)"
