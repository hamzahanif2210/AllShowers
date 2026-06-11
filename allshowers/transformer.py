import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from allshowers.norms import LayerScale, RMSNorm

__all__ = ["FlexEncoderLayer", "Transformer", "compute_mask"]

create_block_mask = torch.compile(create_block_mask)


def compute_mask(
    padding_mask: Tensor,
    layer: Tensor,
    num_layer_cond: int = -1,
) -> BlockMask:
    padding_mask = padding_mask.flatten(1)
    layer = layer.flatten(1)
    if num_layer_cond < 0:

        def mask_fn(b, h, q_idx, kv_idx):
            return padding_mask[b, q_idx] & padding_mask[b, kv_idx]
    else:

        def mask_fn(b, h, q_idx, kv_idx):
            lower_bound = (
                layer[b, q_idx] - layer[b, kv_idx] >= -1 * (num_layer_cond + 1) // 2
            )
            upper_bound = layer[b, q_idx] - layer[b, kv_idx] <= num_layer_cond // 2
            not_padding = padding_mask[b, q_idx] & padding_mask[b, kv_idx]
            return (lower_bound & upper_bound & not_padding) | (q_idx == kv_idx)

    sequence_length = padding_mask.shape[1]
    batch_size = padding_mask.shape[0]
    block_mask = create_block_mask(
        mask_mod=mask_fn,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=str(padding_mask.device),
    )
    return block_mask


class FlexEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        num_head: int = 4,
        dim_feedforward: int = 2048,
        activation: str | torch.nn.Module = "relu",
        dropout: float = 0.0,
        # --- qk / v normalisation ---
        qk_norm: bool = False,
        v_norm: bool = False,
        # --- value residual: passes first-layer value to all blocks ---
        value_residual: bool = False,
        # --- attention gating: sigmoid gate on attn output ---
        attn_gating: bool = False,
        # --- fused MLP: single wide projection instead of two linear layers ---
        fused_mlp: bool = False,
        # --- layer scale init value (None = disabled) ---
        layer_scale_init: float | None = None,
    ) -> None:
        if dim_embedding % num_head != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by num_head ({num_head})."
            )
        super().__init__()

        self.num_head = num_head
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_head
        self.value_residual = value_residual
        self.attn_gating = attn_gating
        self.fused_mlp = fused_mlp

        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation]()
        else:
            activation_module = activation
        del activation

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # QKV projection
        self.key_query_value = nn.Linear(dim_embedding, dim_embedding * 3)

        # Optional per-head Q/K normalisation (applied inside multihead_attention)
        self.q_norm = RMSNorm(self.dim_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dim_head) if qk_norm else nn.Identity()

        # Optional V normalisation
        self.v_norm = RMSNorm(self.dim_head) if v_norm else nn.Identity()

        # Optional attention output gate: x = x * sigmoid(gate(x))
        self.attn_gate = (
            nn.Linear(dim_embedding, dim_embedding) if attn_gating else None
        )

        # Value residual: learned mix of current value and initial value
        # alpha is a per-head scalar initialised to 0 (no residual at start)
        self.value_residual_mix = (
            nn.Parameter(torch.zeros(num_head, 1, self.dim_head))
            if value_residual
            else None
        )

        # MLP / feedforward
        if fused_mlp:
            # Single fused projection: gate * up, then down
            # Uses SwiGLU-style gating: output = (W1x * sigmoid(W2x)) @ W3
            self.mlp_up = nn.Linear(dim_embedding, dim_feedforward * 2)
            self.mlp_down = nn.Linear(dim_feedforward, dim_embedding)
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(dim_embedding, dim_feedforward),
                activation_module,
                nn.Linear(dim_feedforward, dim_embedding),
                self.dropout,
            )

        # Layer norms
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

        # Optional LayerScale on attn and mlp outputs
        self.ls1 = (
            LayerScale(dim_embedding, init_values=layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim_embedding, init_values=layer_scale_init)
            if layer_scale_init is not None
            else nn.Identity()
        )

    def multihead_attention(
        self,
        x: Tensor,
        mask: BlockMask,
        value_init: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (attn_output, value) so value can be passed to later blocks."""
        B, N, _ = x.shape

        key_query_value: Tensor = self.key_query_value(x)
        key_query_value = key_query_value.view(B, N, self.num_head, 3, self.dim_head)
        key_query_value = key_query_value.permute(3, 0, 2, 1, 4).contiguous()
        key, query, value = key_query_value  # each: (B, H, N, dim_head)

        # qk_norm: normalise per head
        query = self.q_norm(query)
        key = self.k_norm(key)

        # v_norm
        value = self.v_norm(value)

        # value_residual: mix current value with the initial (first-block) value
        if self.value_residual and value_init is not None:
            # value_residual_mix is learned, sigmoid-gated
            mix = torch.sigmoid(self.value_residual_mix)  # (H, 1, dim_head)
            value = value * (1.0 - mix) + value_init * mix

        out = flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=mask,
        )  # type: ignore  # (B, H, N, dim_head)

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, self.dim_embedding)
        out = self.dropout(out)

        # attn_gating: element-wise sigmoid gate
        if self.attn_gate is not None:
            out = out * torch.sigmoid(self.attn_gate(x))

        # return value so Transformer can pass it to subsequent blocks
        return out, value

    def mlp(self, x: Tensor) -> Tensor:
        if self.fused_mlp:
            # SwiGLU: split into gate and up halves
            gate, up = self.mlp_up(x).chunk(2, dim=-1)
            x = self.mlp_down(torch.nn.functional.silu(gate) * up)
            return self.dropout(x)
        return self.feedforward(x)

    def forward(
        self,
        x: Tensor,
        mask: BlockMask,
        value_init: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (x, value) where value is the raw pre-attention value tensor
        (shape B, H, N, dim_head) for use as value_init in subsequent blocks.
        """
        attn_out, value = self.multihead_attention(
            self.layer_norm1(x), mask=mask, value_init=value_init
        )
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.layer_norm2(x)))
        return x, value


class Transformer(nn.Module):
    def __init__(
        self,
        dim_inputs: tuple[int, ...],
        dim_embedding: int,
        num_head: int,
        num_blocks: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        num_points_cond: int = 0,
        identity_init: bool = False,
        activation: str | torch.nn.Module = "relu",
        num_layer_cond: int = -1,
        num_particles: int = 1,
        dropout: float = 0.0,
        # --- new feature flags (all off by default) ---
        qk_norm: bool = False,
        v_norm: bool = False,
        value_residual: bool = False,
        attn_gating: bool = False,
        fused_mlp: bool = False,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.num_layer_cond = num_layer_cond
        self.embedding = nn.Linear(dim_inputs[0], dim_embedding)
        self.layer_embedding = nn.Embedding(num_layers, dim_embedding)
        self.cond_embedding = nn.Linear(sum(dim_inputs[1:]), dim_embedding)

        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation.lower()]()
        else:
            activation_module = activation
        del activation

        if num_points_cond > 0:
            self.num_points_embedding = nn.Sequential(
                nn.Linear(num_layers, num_points_cond),
                activation_module,
                nn.Linear(num_points_cond, dim_embedding),
            )
        else:
            self.num_points_embedding = None

        if num_particles > 1:
            self.particle_embedding = nn.Embedding(num_particles, dim_embedding)
        else:
            self.particle_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                FlexEncoderLayer(
                    dim_embedding,
                    num_head,
                    dim_feedforward=dim_feedforward,
                    activation=activation_module,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    v_norm=v_norm,
                    value_residual=value_residual,
                    attn_gating=attn_gating,
                    fused_mlp=fused_mlp,
                    layer_scale_init=layer_scale_init,
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.head = nn.Linear(dim_embedding, dim_inputs[0])
        if identity_init:
            with torch.no_grad():
                self.head.weight.fill_(0.0)
                self.head.bias.fill_(0.0)

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor,
        num_points: Tensor,
        layer: Tensor,
        block_mask: BlockMask,
        label: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        x += self.layer_embedding(layer.squeeze())
        cond = torch.cat([t, cond], dim=1)
        cond = self.cond_embedding(cond).unsqueeze(1)
        x += cond
        if label is not None and self.particle_embedding is not None:
            x += self.particle_embedding(label).unsqueeze(1)
        if self.num_points_embedding is not None:
            num_points = self.num_points_embedding(
                num_points.to(torch.get_default_dtype())
            )
            x += num_points.unsqueeze(1)

        # value_residual: first block produces value_init for all subsequent blocks
        value_init: Tensor | None = None
        for i, block in enumerate(self.transformer_blocks):
            x, value = block(x, mask=block_mask, value_init=value_init)
            if i == 0:
                value_init = value  # capture first-block value

        return self.head(x)