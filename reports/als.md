# Step 3: implicit ALS with fold-in

> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate the pipeline and compare methods on the same data; they are not real-world performance.

ALS is trained on the booking x service matrix (train split). Test bookings are new carts, so each one is **folded in**: its partial cart is solved against the trained service factors with a weighted least-squares step (`movrec.als.fold_in`). Same split, same test cases and same tuning rules as step 2.

- **als**: ALS scores only. An empty cart folds in to the zero vector, so it falls back to global popularity there.
- **als_plus_segment_prior**: ALS scores + w x segment popularity (move_type x home_size), w tuned on val.

Tuning is joint over factors, regularization, alpha and prior weight (each ALS config is fit once; every prior weight is scored from that fit). The grids were widened once, and the search made joint, after a first run picked grid edges with a two-stage search; choices are still made on val only.

## Targets: all 45 services

### Validation tuning (top 5 per model)

| model | factors | reg | alpha | prior_weight | map@5 |
|---|---|---|---|---|---|
| als_plus_segment_prior | 16 | 100 | 10 | 3.0000 | 0.3325 |
| als_plus_segment_prior | 16 | 10 | 10 | 10.0000 | 0.3312 |
| als_plus_segment_prior | 16 | 1 | 10 | 10.0000 | 0.3309 |
| als_plus_segment_prior | 8 | 100 | 10 | 3.0000 | 0.3309 |
| als_plus_segment_prior | 16 | 100 | 10 | 10.0000 | 0.3308 |
| als | 16 | 1000 | 40 | 0.0000 | 0.2115 |
| als | 4 | 1000 | 40 | 0.0000 | 0.2111 |
| als | 8 | 1000 | 40 | 0.0000 | 0.2111 |
| als | 16 | 1000 | 10 | 0.0000 | 0.2080 |
| als | 8 | 1000 | 10 | 0.0000 | 0.2080 |

- Chosen for **als**: factors=16, reg=1000, alpha=40, prior weight=0.0.
- Chosen for **als_plus_segment_prior**: factors=16, reg=100, alpha=10, prior weight=3.0.

### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.704 | 0.798 | 0.428 | 0.334 | - | - |
| cooccurrence | 0.704 | 0.802 | 0.431 | 0.337 | +0.003 (+0.002 to +0.004) | - |
| als | 0.553 | 0.689 | 0.332 | 0.217 | -0.117 (-0.122 to -0.113) | 0.217-0.217 |
| als_plus_segment_prior | 0.710 | 0.810 | 0.438 | 0.343 | +0.009 (+0.007 to +0.011) | 0.342-0.343 |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3064) | cart=1 (n=2654) | cart=2 (n=2358) | cart=3+ (n=5711) |
|---|---|---|---|---|
| segment_popularity | 0.418 / 0.916 | 0.351 / 0.841 | 0.324 / 0.790 | 0.285 / 0.718 |
| cooccurrence | 0.418 / 0.916 | 0.353 / 0.844 | 0.327 / 0.796 | 0.290 / 0.724 |
| als | 0.240 / 0.743 | 0.228 / 0.724 | 0.217 / 0.698 | 0.199 / 0.639 |
| als_plus_segment_prior | 0.418 / 0.916 | 0.357 / 0.849 | 0.328 / 0.803 | 0.302 / 0.738 |

## Targets: add-ons only, truck excluded

### Validation tuning (top 5 per model)

| model | factors | reg | alpha | prior_weight | map@5 |
|---|---|---|---|---|---|
| als_plus_segment_prior | 16 | 100 | 10 | 3.0000 | 0.2374 |
| als_plus_segment_prior | 8 | 10 | 10 | 10.0000 | 0.2368 |
| als_plus_segment_prior | 8 | 1 | 10 | 10.0000 | 0.2367 |
| als_plus_segment_prior | 8 | 100 | 10 | 10.0000 | 0.2364 |
| als_plus_segment_prior | 8 | 100 | 40 | 10.0000 | 0.2360 |
| als | 8 | 1000 | 10 | 0.0000 | 0.2031 |
| als | 4 | 1000 | 10 | 0.0000 | 0.2031 |
| als | 16 | 1000 | 10 | 0.0000 | 0.2031 |
| als | 2 | 1000 | 10 | 0.0000 | 0.2031 |
| als | 16 | 1000 | 40 | 0.0000 | 0.2018 |

- Chosen for **als**: factors=2, reg=1000, alpha=10, prior weight=0.0.
- Chosen for **als_plus_segment_prior**: factors=16, reg=100, alpha=10, prior weight=3.0.

### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.553 | 0.692 | 0.377 | 0.240 | - | - |
| cooccurrence | 0.554 | 0.692 | 0.376 | 0.241 | +0.000 (-0.001 to +0.001) | - |
| als | 0.491 | 0.616 | 0.316 | 0.201 | -0.039 (-0.043 to -0.036) | 0.201-0.201 |
| als_plus_segment_prior | 0.555 | 0.690 | 0.376 | 0.241 | +0.001 (-0.002 to +0.004) | 0.241-0.243 |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3661) | cart=1 (n=2953) | cart=2 (n=2141) | cart=3+ (n=4723) |
|---|---|---|---|---|
| segment_popularity | 0.238 / 0.710 | 0.237 / 0.704 | 0.242 / 0.691 | 0.244 / 0.670 |
| cooccurrence | 0.238 / 0.710 | 0.237 / 0.701 | 0.243 / 0.696 | 0.244 / 0.671 |
| als | 0.195 / 0.607 | 0.198 / 0.627 | 0.204 / 0.625 | 0.207 / 0.613 |
| als_plus_segment_prior | 0.238 / 0.710 | 0.239 / 0.699 | 0.246 / 0.692 | 0.243 / 0.669 |
