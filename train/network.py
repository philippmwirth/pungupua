import haiku as hk
import jax
import jax.numpy as jnp


class ResBlock(hk.Module):
    def __init__(self, channels: int, name: str | None = None):
        super().__init__(name=name)
        self.channels = channels

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        residual = x
        x = hk.Conv2D(self.channels, kernel_shape=3, padding="SAME", with_bias=False)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.channels, kernel_shape=3, padding="SAME", with_bias=False)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
        return jax.nn.relu(x + residual)


class AZNet(hk.Module):
    """Conv residual tower with separate policy and value heads."""

    def __init__(
        self,
        num_actions: int,
        num_channels: int,
        num_blocks: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks

    def __call__(self, x: jnp.ndarray, is_training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, 4, 8, 67)
        x = hk.Conv2D(self.num_channels, kernel_shape=3, padding="SAME", with_bias=False)(x)
        x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
        x = jax.nn.relu(x)

        for i in range(self.num_blocks):
            x = ResBlock(self.num_channels, name=f"res_block_{i}")(x, is_training)

        # Global average pool: (batch, 4, 8, C) -> (batch, C)
        torso = jnp.mean(x, axis=(1, 2))

        # Policy head: (batch, C) -> (batch, num_actions)
        policy = hk.Linear(self.num_channels)(torso)
        policy = jax.nn.relu(policy)
        policy = hk.Linear(self.num_actions)(policy)

        # Value head: (batch, C) -> (batch,)
        value = hk.Linear(self.num_channels)(torso)
        value = jax.nn.relu(value)
        value = hk.Linear(1)(value)
        value = jnp.tanh(value).squeeze(-1)

        return policy, value
