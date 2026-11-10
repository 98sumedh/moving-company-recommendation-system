# Step 5: hybrid

> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate the pipeline and compare methods on the same data; they are not real-world performance.

score(j) = w_seg x segment popularity + w_knn x content kNN + w_als x ALS fold-in + w_cooc x co-occurrence. Component settings come from steps 2-4; only the blend weights are tuned here, on val, by grid search (seg: [0, 1], knn: [0, 0.5, 1, 2, 4, 8], als: [0, 0.1, 0.3, 1, 3], cooc: [0, 0.3, 1, 3]).

- **hybrid_global**: one weight set for every cart.
- **hybrid_by_cart_size**: a separate weight set for each partial-cart size (0, 1, 2, 3+).

Rankings depend only on weight ratios, so `seg` = 1 at its top value is not a search limit. The knn and als grids were widened once after the first run hit their top values. Remaining edge flags ignore that only ratios matter: knn = 8, als = 3, cooc = 0.3 with seg = 0 ranks the same as knn = 1, als = 0.375, cooc = 0.04, i.e. it asks for less co-occurrence, not more knn. The grids were not widened again, to avoid tuning to noise.

## Targets: all 45 services

### Chosen weights (validation)

| model | cart size | seg | knn | als | cooc |
|---|---|---|---|---|---|
| hybrid_global | all | 0 | 1.00 | 0.30 | 0.30 |
| hybrid_by_cart_size | 0 | 1 | 1.00 | 0.00 | 0.00 |
| hybrid_by_cart_size | 1 | 1 | 1.00 | 0.10 | 0.30 |
| hybrid_by_cart_size | 2 | 0 | 0.50 | 0.10 | 0.00 |
| hybrid_by_cart_size | 3+ | 0 | 1.00 | 0.30 | 0.30 |


### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.704 | 0.798 | 0.428 | 0.334 | - | - |
| als_plus_segment_prior | 0.710 | 0.810 | 0.438 | 0.343 | +0.009 (+0.007 to +0.011) | - |
| content_knn | 0.702 | 0.800 | 0.432 | 0.337 | +0.003 (+0.001 to +0.005) | - |
| hybrid_global | 0.711 | 0.811 | 0.438 | 0.343 | +0.010 (+0.007 to +0.012) | - |
| hybrid_by_cart_size | 0.715 | 0.810 | 0.439 | 0.345 | +0.011 (+0.009 to +0.013) | - |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3064) | cart=1 (n=2654) | cart=2 (n=2358) | cart=3+ (n=5711) |
|---|---|---|---|---|
| segment_popularity | 0.418 / 0.916 | 0.351 / 0.841 | 0.324 / 0.790 | 0.285 / 0.718 |
| als_plus_segment_prior | 0.418 / 0.916 | 0.357 / 0.849 | 0.328 / 0.803 | 0.302 / 0.738 |
| content_knn | 0.418 / 0.917 | 0.353 / 0.837 | 0.329 / 0.796 | 0.290 / 0.721 |
| hybrid_global | 0.418 / 0.917 | 0.354 / 0.847 | 0.332 / 0.809 | 0.303 / 0.739 |
| hybrid_by_cart_size | 0.420 / 0.913 | 0.355 / 0.845 | 0.334 / 0.809 | 0.303 / 0.739 |

- hybrid_global vs als_plus_segment_prior (best single model so far): map@5 +0.001 (95% CI -0.001 to +0.003)
- hybrid_by_cart_size vs als_plus_segment_prior (best single model so far): map@5 +0.002 (95% CI +0.000 to +0.004)

### Drop-one-component ablation (test; weights re-tuned on val without the component)

| without | map@5 | change vs full hybrid | 95% CI low | 95% CI high |
|---|---|---|---|---|
| seg | +0.3442 | -0.0005 | -0.0014 | +0.0004 |
| knn | +0.3434 | -0.0012 | -0.0028 | +0.0002 |
| als | +0.3388 | -0.0059 | -0.0074 | -0.0040 |
| cooc | +0.3451 | +0.0004 | -0.0005 | +0.0015 |

## Targets: add-ons only, truck excluded

### Chosen weights (validation)

| model | cart size | seg | knn | als | cooc |
|---|---|---|---|---|---|
| hybrid_global | all | 1 | 4 | 1 | 1.00 |
| hybrid_by_cart_size | 0 | 1 | 1 | 0 | 0.00 |
| hybrid_by_cart_size | 1 | 1 | 4 | 1 | 1.00 |
| hybrid_by_cart_size | 2 | 0 | 8 | 3 | 0.30 |
| hybrid_by_cart_size | 3+ | 1 | 8 | 3 | 3.00 |

- hybrid_by_cart_size: weight at the top of its grid for 2:als, 2:knn, 3+:als, 3+:cooc, 3+:knn (search may be too narrow there).

### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.553 | 0.692 | 0.377 | 0.240 | - | - |
| als_plus_segment_prior | 0.555 | 0.690 | 0.376 | 0.241 | +0.001 (-0.002 to +0.004) | - |
| content_knn | 0.553 | 0.695 | 0.377 | 0.241 | +0.001 (-0.001 to +0.004) | - |
| hybrid_global | 0.560 | 0.701 | 0.384 | 0.246 | +0.006 (+0.003 to +0.008) | - |
| hybrid_by_cart_size | 0.558 | 0.698 | 0.382 | 0.245 | +0.004 (+0.002 to +0.007) | - |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3661) | cart=1 (n=2953) | cart=2 (n=2141) | cart=3+ (n=4723) |
|---|---|---|---|---|
| segment_popularity | 0.238 / 0.710 | 0.237 / 0.704 | 0.242 / 0.691 | 0.244 / 0.670 |
| als_plus_segment_prior | 0.238 / 0.710 | 0.239 / 0.699 | 0.246 / 0.692 | 0.243 / 0.669 |
| content_knn | 0.240 / 0.712 | 0.237 / 0.706 | 0.238 / 0.699 | 0.246 / 0.674 |
| hybrid_global | 0.242 / 0.714 | 0.243 / 0.711 | 0.250 / 0.709 | 0.250 / 0.681 |
| hybrid_by_cart_size | 0.245 / 0.709 | 0.243 / 0.711 | 0.247 / 0.704 | 0.245 / 0.680 |

- hybrid_global vs als_plus_segment_prior (best single model so far): map@5 +0.005 (95% CI +0.003 to +0.007)
- hybrid_by_cart_size vs als_plus_segment_prior (best single model so far): map@5 +0.004 (95% CI +0.002 to +0.005)

### Drop-one-component ablation (test; weights re-tuned on val without the component)

| without | map@5 | change vs full hybrid | 95% CI low | 95% CI high |
|---|---|---|---|---|
| seg | +0.2434 | -0.0015 | -0.0027 | -0.0004 |
| knn | +0.2416 | -0.0033 | -0.0053 | -0.0013 |
| als | +0.2437 | -0.0012 | -0.0039 | +0.0010 |
| cooc | +0.2460 | +0.0011 | -0.0001 | +0.0024 |
