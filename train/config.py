from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    max_num_iters: int = 500

    # network
    num_channels: int = 64
    num_blocks: int = 4

    # self-play
    selfplay_batch_size: int = 32
    num_simulations: int = 16
    max_num_steps: int = 200

    # training
    training_batch_size: int = 32
    learning_rate: float = 1e-3

    # eval
    eval_interval: int = 10
    use_wandb: bool = True

    # resuming
    resume_from: str | None = None
