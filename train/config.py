from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    max_num_iters: int = 100

    # network
    num_channels: int = 128
    num_blocks: int = 6

    # self-play
    selfplay_batch_size: int = 256
    num_simulations: int = 32
    max_num_steps: int = 200

    # training
    training_batch_size: int = 256
    learning_rate: float = 1e-3

    # eval
    eval_interval: int = 10
    use_wandb: bool = True
