# Pungupua

![Logo](assets/pungupua_logo.jpeg)

## Papers

| Paper | Authors | Year | Relevance |
|---|---|---|---|
| [Mastering the Game of Go without Human Knowledge](https://arxiv.org/abs/1712.01815) | Silver et al. | 2017 | Introduced AlphaGo Zero: self-play with MCTS and a dual-headed ResNet (policy + value) |
| [A General RL Algorithm that Masters Chess, Shogi, and Go Through Self-Play](https://www.science.org/doi/10.1126/science.aar6404) | Silver et al. | 2018 | Generalised AlphaGo Zero into AlphaZero, the core algorithm this project implements |
| [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) | Schrittwieser et al. | 2020 | Introduced MuZero; forms the basis of the `mctx` search library used here |
| [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) | Danihelka et al. | 2022 | Introduced Gumbel MuZero, the specific policy (`mctx.gumbel_muzero_policy`) used in training |
| [Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://arxiv.org/abs/2303.17503) | Koyamada et al. | 2023 | The `pgx` library providing the JAX-native Bao game environment |
