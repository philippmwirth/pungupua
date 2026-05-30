# Pungupua

![Logo](assets/pungupua_logo.jpeg)

## Overview

Pungupua (Swahili: *spotted eagle ray*) is an AlphaZero implementation for [Bao la Kiswahili](http://www.gamecabinet.com/rules/Bao.html), a two-player mancala variant from East Africa. A ResNet-based policy/value network is trained from scratch via self-play and Monte Carlo Tree Search (Gumbel MuZero policy), with no human game data. The entire pipeline — environment, search, and training — runs in JAX, making it easy to scale from a laptop to multi-device hardware.

## Papers

| Paper | Authors | Year | Relevance |
|---|---|---|---|
| [Mastering the Game of Go without Human Knowledge](https://arxiv.org/abs/1712.01815) | Silver et al. | 2017 | Introduced AlphaZero: tabula rasa self-play from random play, achieving superhuman performance in chess, shogi, and Go with no domain knowledge beyond the rules |
| [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) | Schrittwieser et al. | 2020 | Introduced MuZero; forms the basis of the `mctx` search library used here |
| [Policy Improvement by Planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO) | Danihelka et al. | 2022 | Introduced Gumbel MuZero, the specific policy (`mctx.gumbel_muzero_policy`) used in training |
| [Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://arxiv.org/abs/2303.17503) | Koyamada et al. | 2023 | The `pgx` library providing the JAX-native Bao game environment |
