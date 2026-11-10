# Step 2: time split and baselines

> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate the pipeline and compare methods on the same data; they are not real-world performance.

## Split (by booking time)

| split | bookings | booked_from | booked_to | moves_from | moves_to | summer_moves |
|---|---|---|---|---|---|---|
| train | 30758 | 2024-06-04 | 2025-12-31 | 2024-10-01 | 2026-04-21 | 27% |
| val | 5428 | 2026-01-01 | 2026-03-31 | 2026-01-01 | 2026-07-25 | 1% |
| test | 13814 | 2026-04-01 | 2026-09-29 | 2026-04-01 | 2026-09-30 | 60% |

Tuning uses val only (models fit on train). Final models are refit on train+val and scored once on test. Test is the summer 2026 peak; val is winter/spring, so tuning and testing happen in different seasons on purpose.

**Protocol:** one case per booking with at least one target item. A random subset of size k ~ Uniform{0..n-1} is revealed as the partial cart; the rest is hidden. Items already in the cart are never recommended. Segment = move_type x home_size (observable, not the hidden segment). The co-occurrence prior-weight grid was widened once after the first run chose its top edge; the choice is still made on val only.

## Targets: all 45 services

### Validation tuning

| model | param | map@5 |
|---|---|---|
| segment_popularity | alpha=0 | 0.3240 |
| segment_popularity | alpha=5 | 0.3240 |
| segment_popularity | alpha=20 | 0.3240 |
| segment_popularity | alpha=100 | 0.3236 |
| segment_popularity | alpha=500 | 0.3207 |
| cooccurrence | prior_weight=0.01 | 0.2679 |
| cooccurrence | prior_weight=0.1 | 0.2827 |
| cooccurrence | prior_weight=0.3 | 0.3029 |
| cooccurrence | prior_weight=1.0 | 0.3223 |
| cooccurrence | prior_weight=3.0 | 0.3267 |
| cooccurrence | prior_weight=10.0 | 0.3255 |

Chosen: segment alpha = 0, co-occurrence prior weight = 3.0.

### Test (all cases)

| model | cases | hr@3 | recall@3 | hr@5 | recall@5 | map@5 | map@5 95% CI |
|---|---|---|---|---|---|---|---|
| global_popularity | 13787 | 0.523 | 0.206 | 0.676 | 0.321 | 0.208 | 0.204-0.212 |
| segment_popularity | 13787 | 0.704 | 0.320 | 0.798 | 0.428 | 0.334 | 0.329-0.339 |
| cooccurrence | 13787 | 0.704 | 0.321 | 0.802 | 0.431 | 0.337 | 0.332-0.341 |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3064) | cart=1 (n=2654) | cart=2 (n=2358) | cart=3+ (n=5711) |
|---|---|---|---|---|
| global_popularity | 0.240 / 0.743 | 0.213 / 0.709 | 0.203 / 0.659 | 0.191 / 0.631 |
| segment_popularity | 0.418 / 0.916 | 0.351 / 0.841 | 0.324 / 0.790 | 0.285 / 0.718 |
| cooccurrence | 0.418 / 0.916 | 0.353 / 0.844 | 0.327 / 0.796 | 0.290 / 0.724 |

## Targets: add-ons only, truck excluded (42 services)

### Validation tuning

| model | param | map@5 |
|---|---|---|
| segment_popularity | alpha=0 | 0.2320 |
| segment_popularity | alpha=5 | 0.2320 |
| segment_popularity | alpha=20 | 0.2317 |
| segment_popularity | alpha=100 | 0.2313 |
| segment_popularity | alpha=500 | 0.2288 |
| cooccurrence | prior_weight=0.01 | 0.2146 |
| cooccurrence | prior_weight=0.1 | 0.2169 |
| cooccurrence | prior_weight=0.3 | 0.2200 |
| cooccurrence | prior_weight=1.0 | 0.2279 |
| cooccurrence | prior_weight=3.0 | 0.2332 |
| cooccurrence | prior_weight=10.0 | 0.2319 |

Chosen: segment alpha = 5, co-occurrence prior weight = 3.0.

### Test (all cases)

| model | cases | hr@3 | recall@3 | hr@5 | recall@5 | map@5 | map@5 95% CI |
|---|---|---|---|---|---|---|---|
| global_popularity | 13478 | 0.491 | 0.210 | 0.616 | 0.316 | 0.201 | 0.197-0.205 |
| segment_popularity | 13478 | 0.553 | 0.251 | 0.692 | 0.377 | 0.240 | 0.236-0.245 |
| cooccurrence | 13478 | 0.554 | 0.252 | 0.692 | 0.376 | 0.241 | 0.236-0.245 |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3661) | cart=1 (n=2953) | cart=2 (n=2141) | cart=3+ (n=4723) |
|---|---|---|---|---|
| global_popularity | 0.195 / 0.607 | 0.198 / 0.627 | 0.204 / 0.625 | 0.206 / 0.613 |
| segment_popularity | 0.238 / 0.710 | 0.237 / 0.704 | 0.242 / 0.691 | 0.244 / 0.670 |
| cooccurrence | 0.238 / 0.710 | 0.237 / 0.701 | 0.243 / 0.696 | 0.244 / 0.671 |
