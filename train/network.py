# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
#
# The bottleneck / broadcast / pooling residual blocks below are transcribed
# from the network appendix of "Policy Improvement by Planning with Gumbel"
# (Danihelka et al., 2022).  Only obvious transcription artifacts are fixed:
# smart quotes, the late-binding closure for the final-block non-linearity, and
# a default `kernel_shape` so `make_conv` can be called for the init conv.

import functools
from typing import Callable, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Legacy basic residual blocks.
#
# Kept for backward compatibility with checkpoints trained before the Gumbel
# blocks were added.  Used by AZNet whenever no bottleneck/broadcast trunk is
# requested (the default), so existing models still load unchanged.
# ---------------------------------------------------------------------------


class BlockV1(hk.Module):
    def __init__(self, num_channels, name="BlockV1"):
        super(BlockV1, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        return jax.nn.relu(x + i)


class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


# ---------------------------------------------------------------------------
# "Policy Improvement by Planning with Gumbel" network modules.
# ---------------------------------------------------------------------------

Tensor = jnp.ndarray
MakeForwardModule = Callable[[], Callable[[Tensor], Tensor]]


class CallArgs(NamedTuple):
    """Per-call flags threaded through the Gumbel blocks (for batch norm)."""

    is_training: bool
    test_local_stats: bool = False


class BasicBlock(hk.Module):
    """Basic block composed of an inner op, a norm op and a non linearity."""

    def __init__(self, make_inner_op, non_linearity=jax.nn.relu, name="basic"):
        super().__init__(name=name)
        self._op = make_inner_op()
        self._norm = hk.BatchNorm(
            create_scale=False, create_offset=True, decay_rate=0.999, eps=1e-3
        )
        self._non_linearity = non_linearity

    def __call__(self, x: Tensor, call_args: CallArgs):
        x = self._op(x)
        x = self._norm(
            x,
            is_training=call_args.is_training,
            test_local_stats=call_args.test_local_stats,
        )
        return self._non_linearity(x)


class ResBlock(hk.Module):
    r"""Creates a residual block with an optional bottleneck."""

    def __init__(
        self, stack_size: int, make_first_op, make_inner_op, make_last_op, name
    ):
        super().__init__(name=name)
        assert stack_size >= 2
        self._blocks = []
        make_ops = (
            [make_first_op] + [make_inner_op] * (stack_size - 2) + [make_last_op]
        )
        for i, make_op in enumerate(make_ops):
            # The final op in the stack uses an identity non-linearity so the
            # ReLU is applied only after the skip connection is added below.
            non_linearity = (lambda x: x) if i == stack_size - 1 else jax.nn.relu
            self._blocks.append(
                BasicBlock(
                    make_inner_op=make_op,
                    non_linearity=non_linearity,
                    name=f"basic_{i}",
                )
            )

    def __call__(self, x: Tensor, call_args: CallArgs):
        res = x
        for b in self._blocks:
            res = b(res, call_args)
        return jax.nn.relu(x + res)


class BroadcastResBlock(ResBlock):
    """A residual block that broadcasts information across spatial dimensions.

    The block consists of a sequence of three layers:
      - a layer that mixes information across channels, e.g. a 1x1 convolution.
      - a layer that mixes information within each channel, a dense layer.
      - another layer to mix across channels.
    The same set of weights is used for mixing information within each channel.
    """

    def __init__(self, make_mix_channel_op, name):
        def broadcast(x: jnp.ndarray):
            n, h, w, c = x.shape
            # Process all planes at once, applying the same linear layer to each.
            x = x.transpose((0, 3, 1, 2))  # NHWC -> NCHW
            x = x.reshape((n, c, h * w))
            x = hk.Linear(h * w, name="broadcast")(x)
            x = jax.nn.relu(x)
            x = x.reshape((n, c, h, w))
            x = x.transpose((0, 2, 3, 1))  # NCHW -> NHWC
            return x

        super().__init__(
            stack_size=3,
            make_first_op=make_mix_channel_op,
            make_inner_op=lambda: broadcast,
            make_last_op=make_mix_channel_op,
            name=name,
        )


class PoolResBlock(hk.Module):
    """A residual block that pools information across spatial dimensions.

    Two parallel BasicBlocks process the input; the second is reduced to a
    per-channel mean/max plane, mixed across channels, and broadcast back to be
    added to the first before a final BasicBlock and skip connection.
    """

    def __init__(self, make_mix_channel_op: MakeForwardModule, name="pool"):
        super().__init__(name=name)
        self._block = functools.partial(
            BasicBlock, make_inner_op=make_mix_channel_op
        )

    def __call__(self, x: Tensor, call_args: CallArgs):
        a = self._block(non_linearity=jax.nn.relu, name="input_a")(x, call_args)
        b = self._block(non_linearity=jax.nn.relu, name="input_b")(x, call_args)
        b_planes = jnp.concatenate([jnp.mean(b, (1, 2)), jnp.max(b, (1, 2))], -1)
        b_planes = hk.Linear(a.shape[-1], name="mix_channels")(b_planes)
        c = a + b_planes[:, None, None, :]
        x = x + self._block(non_linearity=lambda x: x, name="output")(c, call_args)
        return jax.nn.relu(x)


def make_conv(output_channels: int, kernel_shape: int = 3):
    return functools.partial(
        hk.Conv2D,
        output_channels,
        kernel_shape,
        with_bias=False,
        w_init=hk.initializers.TruncatedNormal(0.01),
    )


def make_network(
    num_layers: int,
    output_channels: int,
    bottleneck_channels: int,
    broadcast_every_n: int,
):
    blocks = [
        BasicBlock(
            make_inner_op=make_conv(output_channels),
            non_linearity=jax.nn.relu,
            name="init_conv",
        )
    ]
    for i in range(num_layers):
        if broadcast_every_n > 0 and i % broadcast_every_n == broadcast_every_n - 1:
            blocks.append(
                BroadcastResBlock(
                    make_mix_channel_op=make_conv(output_channels, kernel_shape=1),
                    name=f"broadcast_{i}",
                )
            )
        elif bottleneck_channels > 0:
            blocks.append(
                ResBlock(
                    stack_size=4,
                    make_first_op=make_conv(bottleneck_channels, kernel_shape=1),
                    make_inner_op=make_conv(bottleneck_channels, kernel_shape=3),
                    make_last_op=make_conv(output_channels, kernel_shape=1),
                    name=f"bottleneck_res_{i}",
                )
            )
        else:
            blocks.append(
                ResBlock(
                    stack_size=2,
                    make_first_op=make_conv(output_channels, kernel_shape=3),
                    make_inner_op=make_conv(output_channels, kernel_shape=3),
                    make_last_op=make_conv(output_channels, kernel_shape=3),
                    name=f"res_{i}",
                )
            )
    return blocks


# ---------------------------------------------------------------------------
# AlphaZero network: shared trunk + policy / value heads.
# ---------------------------------------------------------------------------


class AZNet(hk.Module):
    """AlphaZero NN architecture.

    The trunk is built from the Gumbel-paper residual blocks (``make_network``)
    when ``bottleneck`` or ``broadcast_every_n`` is requested; otherwise it uses
    the legacy basic ``BlockV1``/``BlockV2`` blocks so older checkpoints load
    unchanged.
    """

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        bottleneck: bool = False,
        bottleneck_ratio: int = 4,
        broadcast_every_n: int = 0,
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.bottleneck = bottleneck
        self.bottleneck_channels = (
            max(1, num_channels // bottleneck_ratio) if bottleneck else 0
        )
        self.broadcast_every_n = broadcast_every_n
        self.use_gumbel_trunk = bottleneck or broadcast_every_n > 0
        self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

    def _gumbel_trunk(self, x, is_training, test_local_stats):
        call_args = CallArgs(
            is_training=is_training, test_local_stats=test_local_stats
        )
        blocks = make_network(
            num_layers=self.num_blocks,
            output_channels=self.num_channels,
            bottleneck_channels=self.bottleneck_channels,
            broadcast_every_n=self.broadcast_every_n,
        )
        for b in blocks:
            x = b(x, call_args)
        return x

    def _legacy_trunk(self, x, is_training, test_local_stats):
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        if not self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                x, is_training, test_local_stats
            )
        if self.resnet_v2:
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
        return x

    def __call__(self, x, is_training, test_local_stats):
        x = x.astype(jnp.float32)

        if self.use_gumbel_trunk:
            x = self._gumbel_trunk(x, is_training, test_local_stats)
        else:
            x = self._legacy_trunk(x, is_training, test_local_stats)

        # policy head
        logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
        logits = jax.nn.relu(logits)
        logits = hk.Flatten()(logits)
        logits = hk.Linear(self.num_actions)(logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v
