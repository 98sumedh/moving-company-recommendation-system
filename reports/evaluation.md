# Step 6: evaluation

> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate the pipeline and compare methods on the same data; they are not real-world performance.

Every model from steps 2-5 is rebuilt with its chosen settings, fit on train+val, and scored on the same test partial carts (bookings made Apr-Sep 2026, the summer peak). Items already in the cart are never recommended. **cat_hr@k** counts a hit when a top-k item is in the same category as a hidden one (for example, the 25 in wardrobe box when the customer bought the 20 in).

**Ceilings (diagnostics, not models).** `ceiling_true_segment` ranks by an approximation of the generator's own probabilities, including the hidden segment that no model sees. `ceiling_profile_only` averages the hidden segment out with the generator's segment probabilities, so it uses only what a model could know. Both are approximate (see `movrec/oracle.py`), so they are reference points, not exact bounds. **Headroom captured** = (model - segment popularity) / (profile-only ceiling - segment popularity), shown only for models above segment popularity.

**The profile-only ceiling is not a strict upper bound.** It ignores what a revealed cart says about the customer beyond its co-occurrence lifts (a fuller cart hints at a bigger spender, through the generator's shared per-booking noise), and it ignores the random add/drop noise. Models that read the cart (ALS) can pick some of that up, so a model can match or beat it in places. Read it as 'roughly what the observable profile is worth', and the gap to the true-segment ceiling as 'roughly what the hidden segment is worth'.

## Targets: all 45 services

### All models (test)

| model | hr@3 | recall@3 | hr@5 | recall@5 | map@5 | cat_hr@3 | cat_hr@5 | vs segment pop | 95% CI | headroom captured |
|---|---|---|---|---|---|---|---|---|---|---|
| global_popularity | 0.523 | 0.206 | 0.676 | 0.321 | 0.208 | 0.744 | 0.838 | -0.125 | -0.131 to -0.121 | - |
| segment_popularity | 0.704 | 0.320 | 0.798 | 0.428 | 0.334 | 0.834 | 0.884 | 0.000 | +0.000 to +0.000 | - |
| cooccurrence | 0.704 | 0.321 | 0.802 | 0.431 | 0.337 | 0.840 | 0.886 | 0.003 | +0.002 to +0.004 | 14% |
| als | 0.553 | 0.223 | 0.689 | 0.332 | 0.217 | 0.738 | 0.835 | -0.117 | -0.122 to -0.113 | - |
| als_plus_segment_prior | 0.710 | 0.328 | 0.810 | 0.438 | 0.343 | 0.846 | 0.898 | 0.009 | +0.007 to +0.011 | 42% |
| content_knn_equal_weights | 0.660 | 0.290 | 0.768 | 0.403 | 0.296 | 0.795 | 0.869 | -0.038 | -0.041 to -0.034 | - |
| content_knn | 0.702 | 0.319 | 0.800 | 0.432 | 0.337 | 0.837 | 0.893 | 0.003 | +0.001 to +0.005 | 16% |
| hybrid_global | 0.711 | 0.329 | 0.811 | 0.438 | 0.343 | 0.848 | 0.899 | 0.010 | +0.007 to +0.012 | 45% |
| hybrid_by_cart_size | 0.715 | 0.330 | 0.810 | 0.439 | 0.345 | 0.848 | 0.898 | 0.011 | +0.009 to +0.013 | 51% |
| ceiling_profile_only | 0.738 | 0.341 | 0.830 | 0.449 | 0.355 | 0.853 | 0.914 | 0.021 | +0.019 to +0.024 | 100% |
| ceiling_true_segment | 0.745 | 0.352 | 0.836 | 0.464 | 0.371 | 0.862 | 0.916 | 0.037 | +0.034 to +0.040 | 173% |

### map@5 by partial-cart size

| model | cart=all | cart=0 | cart=1 | cart=2 | cart=3+ |
|---|---|---|---|---|---|
| segment_popularity | 0.334 | 0.418 | 0.351 | 0.324 | 0.285 |
| als_plus_segment_prior | 0.343 | 0.418 | 0.357 | 0.328 | 0.302 |
| content_knn | 0.337 | 0.418 | 0.353 | 0.329 | 0.290 |
| hybrid_global | 0.343 | 0.418 | 0.354 | 0.332 | 0.303 |
| ceiling_profile_only | 0.355 | 0.417 | 0.361 | 0.342 | 0.325 |
| ceiling_true_segment | 0.371 | 0.432 | 0.375 | 0.359 | 0.341 |

### map@5 by move type

| model | commercial_office (n=1118) | labor_only (n=1690) | local_residential (n=8529) | long_distance (n=1241) | single_item (n=1209) |
|---|---|---|---|---|---|
| segment_popularity | 0.439 | 0.252 | 0.313 | 0.311 | 0.520 |
| als_plus_segment_prior | 0.444 | 0.257 | 0.324 | 0.320 | 0.527 |
| content_knn | 0.440 | 0.246 | 0.319 | 0.309 | 0.528 |
| hybrid_global | 0.443 | 0.249 | 0.328 | 0.319 | 0.518 |
| ceiling_profile_only | 0.449 | 0.260 | 0.340 | 0.322 | 0.539 |
| ceiling_true_segment | 0.457 | 0.267 | 0.356 | 0.354 | 0.556 |

### Robustness to the reveal seed

Same fitted models, different random partial carts.

| reveal seed | segment_popularity | als_plus_segment_prior | hybrid_global | hybrid_global vs segment pop |
|---|---|---|---|---|
| 2037 | 0.334 | 0.342 | 0.344 | +0.010 (+0.008 to +0.012) |
| 2038 | 0.334 | 0.342 | 0.344 | +0.010 (+0.008 to +0.012) |
| 2039 | 0.333 | 0.340 | 0.341 | +0.008 (+0.006 to +0.010) |

### Ablations (recap)

Feature groups for content kNN (validation, equal weights; from step 4):

| features | map@5 | map@5 (empty cart) |
|---|---|---|
| all groups | 0.2886 | 0.3754 |
| without core | 0.2604 | 0.3301 |
| without access | 0.2886 | 0.3835 |
| without route | 0.2869 | 0.3737 |
| without calendar | 0.3000 | 0.3918 |
| without quote | 0.2768 | 0.3588 |
| core only | 0.3235 | 0.4111 |
| (segment popularity, for reference) | 0.3240 | 0.4093 |

Hybrid components, dropped one at a time and re-tuned on val (test; from step 5):

| without | map@5 | change vs full hybrid | 95% CI low | 95% CI high |
|---|---|---|---|---|
| seg | +0.3442 | -0.0005 | -0.0014 | +0.0004 |
| knn | +0.3434 | -0.0012 | -0.0028 | +0.0002 |
| als | +0.3388 | -0.0059 | -0.0074 | -0.0040 |
| cooc | +0.3451 | +0.0004 | -0.0005 | +0.0015 |

## Targets: add-ons only, truck excluded (42 services)

### All models (test)

| model | hr@3 | recall@3 | hr@5 | recall@5 | map@5 | cat_hr@3 | cat_hr@5 | vs segment pop | 95% CI | headroom captured |
|---|---|---|---|---|---|---|---|---|---|---|
| global_popularity | 0.491 | 0.210 | 0.616 | 0.316 | 0.201 | 0.677 | 0.729 | -0.039 | -0.043 to -0.036 | - |
| segment_popularity | 0.553 | 0.251 | 0.692 | 0.377 | 0.240 | 0.714 | 0.805 | 0.000 | +0.000 to +0.000 | - |
| cooccurrence | 0.554 | 0.252 | 0.692 | 0.376 | 0.241 | 0.718 | 0.804 | 0.000 | -0.001 to +0.001 | 4% |
| als | 0.491 | 0.210 | 0.616 | 0.316 | 0.201 | 0.677 | 0.729 | -0.039 | -0.043 to -0.036 | - |
| als_plus_segment_prior | 0.555 | 0.252 | 0.690 | 0.376 | 0.241 | 0.718 | 0.812 | 0.001 | -0.002 to +0.004 | 15% |
| content_knn_equal_weights | 0.529 | 0.235 | 0.670 | 0.359 | 0.225 | 0.694 | 0.794 | -0.015 | -0.019 to -0.011 | - |
| content_knn | 0.553 | 0.251 | 0.695 | 0.377 | 0.241 | 0.718 | 0.816 | 0.001 | -0.002 to +0.003 | 16% |
| hybrid_global | 0.560 | 0.257 | 0.701 | 0.384 | 0.246 | 0.718 | 0.821 | 0.006 | +0.003 to +0.008 | 98% |
| hybrid_by_cart_size | 0.558 | 0.254 | 0.698 | 0.382 | 0.245 | 0.719 | 0.822 | 0.004 | +0.002 to +0.007 | 74% |
| ceiling_profile_only | 0.565 | 0.255 | 0.701 | 0.377 | 0.246 | 0.719 | 0.837 | 0.006 | +0.003 to +0.010 | 100% |
| ceiling_true_segment | 0.588 | 0.271 | 0.720 | 0.397 | 0.266 | 0.740 | 0.842 | 0.025 | +0.022 to +0.029 | 419% |

### map@5 by partial-cart size

| model | cart=all | cart=0 | cart=1 | cart=2 | cart=3+ |
|---|---|---|---|---|---|
| segment_popularity | 0.240 | 0.238 | 0.237 | 0.242 | 0.244 |
| als_plus_segment_prior | 0.241 | 0.238 | 0.239 | 0.246 | 0.243 |
| content_knn | 0.241 | 0.240 | 0.237 | 0.238 | 0.246 |
| hybrid_global | 0.246 | 0.242 | 0.243 | 0.250 | 0.250 |
| ceiling_profile_only | 0.246 | 0.239 | 0.241 | 0.248 | 0.255 |
| ceiling_true_segment | 0.266 | 0.264 | 0.260 | 0.267 | 0.270 |

### map@5 by move type

| model | commercial_office (n=1112) | labor_only (n=1690) | local_residential (n=8354) | long_distance (n=1236) | single_item (n=1086) |
|---|---|---|---|---|---|
| segment_popularity | 0.369 | 0.255 | 0.215 | 0.229 | 0.292 |
| als_plus_segment_prior | 0.365 | 0.259 | 0.217 | 0.233 | 0.287 |
| content_knn | 0.369 | 0.247 | 0.217 | 0.228 | 0.304 |
| hybrid_global | 0.369 | 0.253 | 0.224 | 0.234 | 0.298 |
| ceiling_profile_only | 0.358 | 0.264 | 0.221 | 0.238 | 0.312 |
| ceiling_true_segment | 0.366 | 0.269 | 0.243 | 0.263 | 0.338 |

### Robustness to the reveal seed

Same fitted models, different random partial carts.

| reveal seed | segment_popularity | als_plus_segment_prior | hybrid_global | hybrid_global vs segment pop |
|---|---|---|---|---|
| 2037 | 0.239 | 0.240 | 0.246 | +0.007 (+0.004 to +0.010) |
| 2038 | 0.242 | 0.244 | 0.249 | +0.007 (+0.005 to +0.010) |
| 2039 | 0.243 | 0.247 | 0.252 | +0.008 (+0.006 to +0.011) |

### Ablations (recap)

Feature groups for content kNN (validation, equal weights; from step 4):

| features | map@5 | map@5 (empty cart) |
|---|---|---|
| all groups | 0.2237 | 0.2301 |
| without core | 0.2107 | 0.2161 |
| without access | 0.2162 | 0.2263 |
| without route | 0.2224 | 0.2282 |
| without calendar | 0.2288 | 0.2307 |
| without quote | 0.2205 | 0.2266 |
| core only | 0.2297 | 0.2327 |
| (segment popularity, for reference) | 0.2320 | 0.2376 |

Hybrid components, dropped one at a time and re-tuned on val (test; from step 5):

| without | map@5 | change vs full hybrid | 95% CI low | 95% CI high |
|---|---|---|---|---|
| seg | +0.2434 | -0.0015 | -0.0027 | -0.0004 |
| knn | +0.2416 | -0.0033 | -0.0053 | -0.0013 |
| als | +0.2437 | -0.0012 | -0.0039 | +0.0010 |
| cooc | +0.2460 | +0.0011 | -0.0001 | +0.0024 |
