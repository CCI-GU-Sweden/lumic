# Convex hull dimensions: three simulated signatures

150 random mixtures of EBFP2, EGFP, mCherry, approximate 980_32ch detector, seed 42; noise standard deviation = 0.001 × maximum noiseless value. Each case has a 8 s limit including worker startup.

| Data | Matrix rank | npc | Hull dimensions | Outcome | Vertices |
| --- | ---: | ---: | ---: | --- | ---: |
| noiseless | 3 | 3 | 2 | completed | 14 |
| noiseless | 3 | 4 | 3 | qhull_error |  |
| noiseless | 3 | 6 | 5 | qhull_error |  |
| noiseless | 3 | 8 | 7 | qhull_error |  |
| noiseless | 3 | 12 | 11 | qhull_error |  |
| noiseless | 3 | 16 | 15 | qhull_error |  |
| noiseless | 3 | 24 | 23 | qhull_error |  |
| noiseless | 3 | 32 | 31 | qhull_error |  |
| noisy | 32 | 3 | 2 | completed | 13 |
| noisy | 32 | 4 | 3 | completed | 33 |
| noisy | 32 | 6 | 5 | completed | 87 |
| noisy | 32 | 8 | 7 | completed | 125 |
| noisy | 32 | 12 | 11 | timeout |  |
| noisy | 32 | 16 | 15 | timeout |  |
| noisy | 32 | 24 | 23 | timeout |  |
| noisy | 32 | 32 | 31 | timeout |  |

A timeout is not proof that a hull cannot be computed. Noise can create extra numerical dimensions without adding fluorophores. This experiment does not evaluate unmixing or establish a fluorophore limit.
