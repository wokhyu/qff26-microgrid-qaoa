## Solution quality (mean over instances)

QAOA rows report the best of the sampled bitstrings (shots as configured). `objective gap` is the QUBO objective above the exact optimum: generation cost plus the balance penalty, so it is not a cash amount. `surplus` is supply above demand (reserve plus curtailment).

| n | method | p | found optimum | objective gap | % demand met | surplus (kW) | generation cost (USD/h) | runtime (s) |
|---|---|---|---|---|---|---|---|---|
| 6 | exact | - | 100% | 0.00 | 93.4 | 20.2 | 14.37 | 0.00 |
| 6 | greedy | - | 33% | 24.92 | 93.9 | 40.7 | 13.52 | 0.00 |
| 6 | sa | - | 100% | -0.00 | 93.4 | 20.2 | 14.37 | 0.12 |
| 6 | qaoa_ideal | 3 | 100% | 0.00 | 93.4 | 20.2 | 14.37 | 1.61 |
| 6 | qaoa_noisy | 3 | 100% | 0.00 | 93.4 | 20.2 | 14.37 | 7.92 |
| 8 | exact | - | 100% | 0.00 | 94.4 | 32.8 | 16.69 | 0.00 |
| 8 | greedy | - | 22% | 13.97 | 93.4 | 45.7 | 13.10 | 0.00 |
| 8 | sa | - | 100% | -0.00 | 94.4 | 32.8 | 16.69 | 0.15 |
| 8 | qaoa_ideal | 3 | 100% | 0.00 | 94.4 | 32.8 | 16.69 | 1.91 |
| 8 | qaoa_noisy | 3 | 100% | 0.00 | 94.4 | 32.8 | 16.69 | 8.10 |
| 10 | exact | - | 100% | 0.00 | 91.4 | 22.8 | 15.35 | 0.00 |
| 10 | greedy | - | 22% | 10.84 | 91.3 | 39.4 | 14.28 | 0.00 |
| 10 | sa | - | 89% | 0.03 | 91.4 | 22.2 | 15.27 | 0.17 |
| 10 | qaoa_ideal | 3 | 100% | 0.00 | 91.4 | 22.8 | 15.35 | 3.56 |
| 10 | qaoa_noisy | 3 | 100% | 0.00 | 91.4 | 22.8 | 15.35 | 8.43 |
| 12 | exact | - | 100% | 0.00 | 93.7 | 31.2 | 18.37 | 0.00 |
| 12 | greedy | - | 11% | 19.29 | 92.6 | 44.5 | 16.55 | 0.00 |
| 12 | sa | - | 100% | -0.00 | 93.7 | 31.2 | 18.37 | 0.20 |
| 12 | qaoa_ideal | 3 | 100% | 0.00 | 93.7 | 31.2 | 18.37 | 6.13 |
| 12 | qaoa_noisy | 3 | 100% | 0.00 | 93.7 | 31.2 | 18.37 | 37.55 |

## How much QAOA concentrates probability on the optimum

`P(opt)` is the probability that one shot returns an optimal dispatch. `random` is the same for a uniform guess. `amplification` = P(opt) / random. `expectation ratio` is the approximation ratio of the mean energy.

| n | method | p | expectation ratio | P(opt) | random | amplification | 2q depth (transpiled) |
|---|---|---|---|---|---|---|---|
| 6 | qaoa_ideal | 1 | 0.879 | 0.0811 | 0.01562 | 5.2x | 28 |
| 6 | qaoa_ideal | 2 | 0.925 | 0.1724 | 0.01562 | 11.0x | 64 |
| 6 | qaoa_ideal | 3 | 0.946 | 0.2351 | 0.01562 | 15.0x | 97 |
| 6 | qaoa_noisy | 1 | 0.866 | 0.0729 | 0.01562 | 4.7x | 28 |
| 6 | qaoa_noisy | 2 | 0.898 | 0.1378 | 0.01562 | 8.8x | 64 |
| 6 | qaoa_noisy | 3 | 0.908 | 0.1670 | 0.01562 | 10.7x | 97 |
| 8 | qaoa_ideal | 1 | 0.895 | 0.0218 | 0.00391 | 5.6x | 45 |
| 8 | qaoa_ideal | 2 | 0.922 | 0.0458 | 0.00391 | 11.7x | 88 |
| 8 | qaoa_ideal | 3 | 0.942 | 0.0790 | 0.00391 | 20.2x | 137 |
| 8 | qaoa_noisy | 1 | 0.884 | 0.0189 | 0.00391 | 4.8x | 45 |
| 8 | qaoa_noisy | 2 | 0.898 | 0.0326 | 0.00391 | 8.3x | 88 |
| 8 | qaoa_noisy | 3 | 0.907 | 0.0483 | 0.00391 | 12.4x | 137 |
| 10 | qaoa_ideal | 1 | 0.887 | 0.0080 | 0.00098 | 8.2x | 66 |
| 10 | qaoa_ideal | 2 | 0.919 | 0.0203 | 0.00098 | 20.8x | 135 |
| 10 | qaoa_ideal | 3 | 0.936 | 0.0316 | 0.00098 | 32.4x | 201 |
| 10 | qaoa_noisy | 1 | 0.875 | 0.0066 | 0.00098 | 6.7x | 66 |
| 10 | qaoa_noisy | 2 | 0.892 | 0.0132 | 0.00098 | 13.5x | 135 |
| 10 | qaoa_noisy | 3 | 0.894 | 0.0161 | 0.00098 | 16.5x | 201 |
| 12 | qaoa_ideal | 1 | 0.903 | 0.0023 | 0.00024 | 9.6x | 99 |
| 12 | qaoa_ideal | 2 | 0.929 | 0.0054 | 0.00024 | 22.1x | 177 |
| 12 | qaoa_ideal | 3 | 0.945 | 0.0094 | 0.00024 | 38.5x | 230 |
| 12 | qaoa_noisy | 1 | 0.887 | 0.0018 | 0.00024 | 7.5x | 99 |
| 12 | qaoa_noisy | 2 | 0.896 | 0.0030 | 0.00024 | 12.3x | 177 |
| 12 | qaoa_noisy | 3 | 0.898 | 0.0034 | 0.00024 | 13.9x | 230 |

## Shots needed to be 95% sure of sampling the optimal plan (p = 3, mean over instances)

Best-of-shots success is easy at small n, because random guessing also gets lucky. This table is the fair comparison.

| n | random guessing | QAOA noise-free | QAOA Heron noise model |
|---|---|---|---|
| 6 | 190 | 15 | 21 |
| 8 | 765 | 46 | 83 |
| 10 | 3,066 | 133 | 332 |
| 12 | 12,269 | 425 | 1,367 |
