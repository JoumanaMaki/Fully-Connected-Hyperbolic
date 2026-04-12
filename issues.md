# Codebase Issues

## Critical: Dual curvature convention (`-1/k` vs `-k`)

The manifold is documented as `-x_0^2 + x_1^2 + ... = -1/k`, but several functions use the `-k` convention instead. **For k=1 (the default) these are identical, so the bugs are latent.**

| Function | File:Line | Convention | Should be |
|---|---|---|---|
| `projection_space_orthogonal` | lorentz.py:294 | `-1/k` | (reference) |
| `expmap0` | lorentz.py:47 | `-1/k` | (reference) |
| `parallel_transport` | lorentz.py:199 | `-1/k` | (reference) |
| **`origin()`** | lorentz.py:393 | **`-k`** | `-1/k` |
| **`projx`** | lorentz.py:100 | **`-k`** | `-1/k` |
| **`add_time`** | lorentz.py:406 | **`-k`** | `-1/k` |
| **`dist0`** | lorentz.py:442 | **`-k`** | `-1/k` |
| **`logmap0_full`** | lorentz.py:449 | uses wrong origin | `-1/k` |
| **`logmap0back`** | lorentz.py:477 | uses wrong origin | `-1/k` |
| **`proju`** | lorentz.py:108 | **`-k`** | `-1/k` |
| **`LorentzFullyConnectedTheirs`** | LLinear.py:254 | **`-k`** (via `add_time`) | `-1/k` |

`ChenLinear` (chen.py:55) uses the correct `1/k` convention, so `ChenLinear` and `LorentzFullyConnectedTheirs` disagree despite implementing the same paper's method.

`origin()` returns `(sqrt(k), 0, ...)` but `expmap0(zero)` returns `(1/sqrt(k), 0, ...)`. These are only equal at k=1.

## Crash bugs

1. **`Lorentz(k=1.0)` -- wrong keyword argument** in 3 locations. The constructor accepts `k_value`, not `k`. Will raise `TypeError` if the fallback paths execute:
   - `LConv.py:26`
   - `LResBlock.py:27`
   - `toy_exp/toy_exp.py:286`

## Correctness bugs

2. **`LorentzMLR` allocates `num_classes + 1` hyperplanes** (LLinear.py:159-160). Standard MLR needs `num_classes`. The forward method produces one extra logit. Compare with `BdeirLorentzMLR` (bdeir.py) which correctly uses `num_classes`.

3. **`LResBlock` skip connection missing stride check** (LResBlock.py:58). The condition is `if input_dim != output_dim` but should be `if input_dim != output_dim or stride != 1`. When dims match but stride > 1, the skip path uses `nn.Identity()` while the main path downsamples, causing a shape mismatch at the addition on line 87. Currently latent because all ResNet stages with stride > 1 also change dims, but triggers if `embedding_dim` equals `base_dim * 4`.

4. **`ChenLinear` applies ReLU to the full Lorentz point** (chen.py:65). After `projection_space_orthogonal` produces a valid manifold point, `self.activation(x)` zeros out negative space components without recomputing the time component, producing off-manifold points.

5. **`BdeirLorentzMLR` uses inverted curvature** (bdeir.py:32). Sets `sqrt_mK = self.manifold.k().sqrt()` (= `sqrt(k)`) while `LorentzMLR` (LLinear.py:166) uses `1/self.manifold.k().sqrt()` (= `1/sqrt(k)`). Produces wrong results for k != 1.

6. **`LorentzFullyConnectedTheirs` in-place masking breaks autograd** (LLinear.py:250). `square_norm[mask] = 1` mutates a tensor in the computation graph. Should use `torch.where`.

7. **CIFAR StepLR milestone skip wrong with warmup** (cifar_exp/main.py:480). `steplr_first_milestone` (line 383) uses the unadjusted epoch, but when `warmup_epochs > 0` the `SequentialLR` shifts the milestones (line 395). The skip fires at the wrong epoch.

8. **CIFAR "scale" params excluded from all optimizer groups** (cifar_exp/main.py:23,45-51). Params with "scale" in the name match `no_decay` but aren't added to any group, so they're never trained.

9. **Runtime benchmark uses random `V_auxiliary`** (runtime_exp/my_runtimes.py:183). `forward_cache` is called without first calling `compute_V_auxiliary()`, so the benchmark runs on random initialization rather than the actual learned spacelike vectors.

## Config / logic issues

10. **CIFAR default config is self-contradictory** (cifar_exp/main.py:581-627). The coupled parameters `lorentz_method: "theirs"` (line 593) silently overrides `fc_variant: "ours"` (line 592) and `mlr_type: "fc_mlr"` (line 590). Similarly `norm_config: "normal_noweightnorm"` (line 594) overrides `normalisation_mode: "centering_only"` (line 589) and `use_weight_norm: True` (line 626). Individual settings are dead code.

## Numerical stability

11. **`normL` -- no clamp before sqrt** (lorentz.py:30). Called on tangent vectors which should be spacelike, but floating-point noise can make the sum slightly negative, producing NaN.

12. **`lorentz_midpoint` -- no clamp before sqrt** (lorentz.py:374). `lorentz_norm_sq.sqrt()` with no clamp, risking NaN from noisy inputs.

## Minor / cleanup

13. **`proju` and `egrad2rgrad` mutate inputs in-place** (lorentz.py:108, 339). Can silently corrupt caller's tensors.

14. **Deprecated `clip_grad_norm`** (toy_exp.py:204). Should be `clip_grad_norm_` (with trailing underscore).

15. **Unused import** `from einops import rearrange` (LLinear.py:5).

16. **Unused parameter** `constraining_strategy` in `Lorentz.__init__` (lorentz.py:14).

17. **`PoincareActivation` not in `__all__`** (__init__.py). Imported but not exported.

18. **`BdeirLorentzMLR` not importable** from the package (not in __init__.py).

19. **`preactivation_map` hardcodes `manifold_dim=1`** (lorentz.py:38). Would be wrong for non-image tensors.
