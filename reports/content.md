# Step 4: move-profile vectors and cosine nearest neighbours

> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate the pipeline and compare methods on the same data; they are not real-world performance.

Each booking becomes a vector of what is known at booking time: move type and size (**core**), floors, stairs and elevators (**access**), regions and distance (**route**), month, weekday, lead time and arrival window (**calendar**), and estimate method, payment method and crew (**quote**). Categoricals are one-hot, numerics standardized, each group scaled to equal weight, rows L2-normalized. The truck is never a feature.

A new booking is matched to its k most similar past bookings by cosine similarity; services are scored by the similarity-weighted share of neighbours who bought them. The partial cart is not used, so this is the model for **empty carts**. Same split, cases and tuning rules as earlier steps.

The first run used equal group weights and lost to segment popularity. The ablation showed weak groups diluting `core`, so group weights are now tuned on val (both versions are reported on test).

## Targets: all 45 services

### Validation tuning

| weights | k | power | map@5 | map@5 (empty cart) |
|---|---|---|---|---|
| tuned | 300 | 4 | 0.3280 | 0.4145 |
| tuned | 100 | 4 | 0.3257 | 0.4106 |
| tuned | 1000 | 4 | 0.3240 | 0.4091 |
| tuned | 30 | 4 | 0.3104 | 0.3993 |
| equal | 300 | 4 | 0.2886 | 0.3754 |
| equal | 100 | 4 | 0.2877 | 0.3738 |
| equal | 300 | 1 | 0.2874 | 0.3748 |
| equal | 100 | 1 | 0.2870 | 0.3725 |
| tuned | 10 | 4 | 0.2820 | 0.3636 |
| equal | 1000 | 4 | 0.2803 | 0.3672 |
| equal | 30 | 4 | 0.2798 | 0.3547 |
| equal | 30 | 1 | 0.2797 | 0.3555 |
| equal | 1000 | 1 | 0.2786 | 0.3671 |
| equal | 10 | 1 | 0.2525 | 0.3310 |
| equal | 10 | 4 | 0.2525 | 0.3310 |

### Feature-group ablation (validation, equal weights)

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

### Group-weight search (validation)

| step | core | access | route | calendar | quote | map@5 |
|---|---|---|---|---|---|---|
| core only | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3235 |
| set access | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.3276 |
| set route | 1.0000 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.3280 |
| set calendar | 1.0000 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.3280 |
| set quote | 1.0000 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.3280 |

Chosen: equal weights k = 300; tuned weights {'core': 1.0, 'access': 0.25, 'route': 0.25, 'calendar': 0.0, 'quote': 0.0}, k = 300; power = 4.

### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.704 | 0.798 | 0.428 | 0.334 | - | - |
| als_plus_segment_prior | 0.710 | 0.810 | 0.438 | 0.343 | +0.009 (+0.007 to +0.011) | - |
| content_knn_equal_weights | 0.660 | 0.768 | 0.403 | 0.296 | -0.038 (-0.041 to -0.034) | - |
| content_knn | 0.702 | 0.800 | 0.432 | 0.337 | +0.003 (+0.001 to +0.005) | - |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3064) | cart=1 (n=2654) | cart=2 (n=2358) | cart=3+ (n=5711) |
|---|---|---|---|---|
| segment_popularity | 0.418 / 0.916 | 0.351 / 0.841 | 0.324 / 0.790 | 0.285 / 0.718 |
| als_plus_segment_prior | 0.418 / 0.916 | 0.357 / 0.849 | 0.328 / 0.803 | 0.302 / 0.738 |
| content_knn_equal_weights | 0.370 / 0.882 | 0.313 / 0.807 | 0.291 / 0.770 | 0.251 / 0.689 |
| content_knn | 0.418 / 0.917 | 0.353 / 0.837 | 0.329 / 0.796 | 0.290 / 0.721 |

Empty carts only: content kNN vs segment popularity map@5 -0.000 (95% CI -0.004 to +0.004).

## Targets: add-ons only, truck excluded

### Validation tuning

| weights | k | power | map@5 | map@5 (empty cart) |
|---|---|---|---|---|
| tuned | 300 | 1 | 0.2335 | 0.2362 |
| tuned | 1000 | 1 | 0.2312 | 0.2346 |
| tuned | 100 | 1 | 0.2311 | 0.2324 |
| equal | 300 | 1 | 0.2237 | 0.2301 |
| equal | 300 | 4 | 0.2236 | 0.2289 |
| equal | 1000 | 4 | 0.2217 | 0.2320 |
| equal | 1000 | 1 | 0.2211 | 0.2333 |
| equal | 100 | 1 | 0.2177 | 0.2191 |
| equal | 100 | 4 | 0.2176 | 0.2181 |
| tuned | 30 | 1 | 0.2161 | 0.2183 |
| equal | 30 | 1 | 0.2063 | 0.2096 |
| equal | 30 | 4 | 0.2056 | 0.2086 |
| tuned | 10 | 1 | 0.1869 | 0.1878 |
| equal | 10 | 4 | 0.1782 | 0.1853 |
| equal | 10 | 1 | 0.1780 | 0.1852 |

### Feature-group ablation (validation, equal weights)

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

### Group-weight search (validation)

| step | core | access | route | calendar | quote | map@5 |
|---|---|---|---|---|---|---|
| core only | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2297 |
| set access | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.2335 |
| set route | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.2335 |
| set calendar | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.2335 |
| set quote | 1.0000 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.2335 |

Chosen: equal weights k = 300; tuned weights {'core': 1.0, 'access': 0.25, 'route': 0.0, 'calendar': 0.0, 'quote': 0.0}, k = 300; power = 1.

### Test

| model | hr@3 | hr@5 | recall@5 | map@5 | map@5 vs segment pop (95% CI) | seed spread |
|---|---|---|---|---|---|---|
| segment_popularity | 0.553 | 0.692 | 0.377 | 0.240 | - | - |
| als_plus_segment_prior | 0.555 | 0.690 | 0.376 | 0.241 | +0.001 (-0.002 to +0.003) | - |
| content_knn_equal_weights | 0.529 | 0.670 | 0.359 | 0.225 | -0.015 (-0.018 to -0.011) | - |
| content_knn | 0.553 | 0.695 | 0.377 | 0.241 | +0.001 (-0.001 to +0.004) | - |

### Test by partial-cart size (map@5 / hr@5)

| model | cart=0 (n=3661) | cart=1 (n=2953) | cart=2 (n=2141) | cart=3+ (n=4723) |
|---|---|---|---|---|
| segment_popularity | 0.238 / 0.710 | 0.237 / 0.704 | 0.242 / 0.691 | 0.244 / 0.670 |
| als_plus_segment_prior | 0.238 / 0.710 | 0.239 / 0.699 | 0.246 / 0.692 | 0.243 / 0.669 |
| content_knn_equal_weights | 0.227 / 0.686 | 0.223 / 0.679 | 0.224 / 0.670 | 0.227 / 0.653 |
| content_knn | 0.240 / 0.712 | 0.237 / 0.706 | 0.238 / 0.699 | 0.246 / 0.674 |

Empty carts only: content kNN vs segment popularity map@5 +0.002 (95% CI -0.003 to +0.007).
