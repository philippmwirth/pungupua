from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    max_num_iters: int = 500

    # network
    num_channels: int = 64
    num_blocks: int = 2

    # self-play
    selfplay_batch_size: int = 32
    num_simulations: int = 32
    max_num_steps: int = 96
    # Number of opening moves per game during which the played action is sampled
    # proportionally to MCTS visit counts (exploration). After this many moves
    # the agent plays the deterministic action proposed by the search.
    num_sampling_moves: int = 30

    # training
    training_batch_size: int = 1024
    # FIFO replay buffer capacity, measured in positions (samples). Each
    # iteration's self-play data is appended and the oldest positions are
    # evicted once capacity is exceeded; training minibatches are drawn from the
    # whole buffer rather than only the latest round.
    replay_buffer_size: int = 100_000
    learning_rate: float = 1e-3
    learning_rate_min: float = 1e-5
    learning_rate_warmup_steps: int = 100

    # eval
    eval_interval: int = 10
    use_wandb: bool = True

    # resuming
    resume_from: str | None = None
