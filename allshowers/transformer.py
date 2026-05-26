import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

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
        attn_type: str = "flex",
    ) -> None:
        if dim_embedding % num_head != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by num_head ({num_head})."
            )
        super().__init__()

        self.num_head = num_head
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_head
        self.attn_type = attn_type

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

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        if attn_type == "flex":
            self.key_query_value = nn.Linear(dim_embedding, dim_embedding * 3)
            self.fast_attn = None
        else:
            from allshowers.attention import Attention
            self.key_query_value = None
            self.fast_attn = Attention(dim=dim_embedding, num_heads=num_head, attn_type=attn_type)

        self.feedforward = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            activation_module,
            nn.Linear(dim_feedforward, dim_embedding),
            self.dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def multihead_attention(
        self,
        x: Tensor,
        mask: BlockMask,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.fast_attn is not None:
            # flash-varlen and torch use the boolean padding mask; flash ignores masking
            if self.attn_type in ("flash-varlen", "torch"):
                return self.fast_attn(x, kv_mask=padding_mask)
            return self.fast_attn(x)

        # Original flex_attention path
        key_query_value: Tensor = self.key_query_value(x)
        key_query_value = key_query_value.view(
            key_query_value.shape[0],
            key_query_value.shape[1],
            self.num_head,
            3,
            self.dim_head,
        )
        key_query_value = key_query_value.permute(3, 0, 2, 1, 4).contiguous()
        key, query, value = key_query_value

        x = flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=mask,
        )  # type: ignore

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.dim_embedding)
        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        mask: BlockMask,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.multihead_attention(self.layer_norm1(x), mask=mask, padding_mask=padding_mask)
        x = x + self.feedforward(self.layer_norm2(x))
        return x


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
        attn_type: str = "flex",
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
                    attn_type=attn_type,
                )
                for _ in range(num_blocks)
            ]
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

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
        padding_mask: Tensor | None = None,
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
        for block in self.transformer_blocks:
            x = block(x, mask=block_mask, padding_mask=padding_mask)
        return self.head(x)


# ---------------------------------------------------------------------------
# DiT-style transformer components
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Literal

from allshowers.attention import Attention
from allshowers.norms import RMSNorm, LayerNorm


class GLU(nn.Module):
    """Dense update with Gated Linear Unit. See https://arxiv.org/abs/2002.05202."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        activation: str = "SiLU",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
        self.gated = gated
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(embed_dim, hidden_dim + hidden_dim * gated, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

    def modulate(self, x: Tensor) -> Tensor:
        return x * (1 + self.scale) + self.shift


class Modulation(nn.Module):
    """AdaLN-style modulation from a context vector."""

    def __init__(self, dim: int, double: bool, ctxt_ratio: float = 3.0):
        super().__init__()
        self.dim = dim
        self.ctxt_ratio = ctxt_ratio
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(int(dim * ctxt_ratio), int(self.multiplier * dim), bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

    def reset_parameters(self):
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)


class DiTLayer(nn.Module):
    """Single DiT layer: cross-attention (or self-attention) with AdaLN modulation and optional GLU MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mha_config: dict | None = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
        simplify: bool = False,
        norm: Literal["rms", "layer"] = "rms",
    ):
        super().__init__()
        self.simplify = simplify
        if norm == "layer":
            norm_layer = LayerNorm
        elif norm == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm type: {norm}. Supported: 'rms', 'layer'.")
        self.norm1 = norm_layer(hidden_dim)
        self.norm_k = norm_layer(hidden_dim)
        if not simplify:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = GLU(hidden_dim, mlp_hidden_dim, bias=False, gated=True)
        self.adaLN_modulation = Modulation(hidden_dim, double=not simplify, ctxt_ratio=ctxt_ratio)
        if mha_config is None:
            mha_config = {}
        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            fuse_mlp=simplify,
            mlp_ratio=mlp_ratio,
            **mha_config,
        )
        self.set_backend(mha_config.get("attn_type", "torch"))

    def set_backend(self, attn_type: str):
        self.attn.set_backend(attn_type)
        self.attn_type = attn_type
        return self.attn_type

    def forward(
        self,
        seq_q: Tensor,
        seq_k: Tensor,
        ctxt: Tensor,
        mask_k: Tensor | BlockMask | None = None,
    ):
        mod_mha, mod_mlp = self.adaLN_modulation(ctxt)
        seq_q_mod = mod_mha.modulate(self.norm1(seq_q))
        seq_k = self.norm_k(seq_k)

        if self.simplify:
            return seq_q + mod_mha.gate * self.attn(
                q=seq_q_mod,
                kv=seq_k,
                kv_mask=mask_k if self.attn_type != "flex" else None,
                attn_mask=mask_k if self.attn_type == "flex" else None,
            )

        seq_q = seq_q + mod_mha.gate * self.attn(
            q=seq_q_mod,
            kv=seq_k,
            kv_mask=mask_k if self.attn_type != "flex" else None,
            attn_mask=mask_k if self.attn_type == "flex" else None,
        )
        seq_q = seq_q + mod_mlp.gate * self.mlp(mod_mlp.modulate(self.norm2(seq_q)))
        return seq_q

    def reset_parameters(self):
        self.adaLN_modulation.reset_parameters()


class SADiTLayer(DiTLayer):
    """DiTLayer wired for self-attention (seq_q == seq_k)."""

    def forward(self, seq: Tensor, ctxt: Tensor, mask: Tensor | BlockMask | None = None):
        return super().forward(seq, seq, ctxt, mask)


class CADiTLayer(DiTLayer):
    """DiTLayer wired for cross-attention."""

    def forward(
        self,
        seq_q: Tensor,
        seq_k: Tensor,
        ctxt: Tensor,
        mask_k: Tensor | BlockMask | None = None,
    ):
        return super().forward(seq_q, seq_k, ctxt, mask_k)


class DiTBlock(nn.Module):
    """
    One block of the DiT encoder.

    Optionally bidirectional (seq_a attends to seq_b AND seq_b attends to seq_a)
    and optionally prefixed by self-attention on each sequence.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        mha_config=None,
        mlp_ratio=4.0,
        ctxt_ratio=3.0,
        bidirectional=False,
        do_selfattn=False,
        simplify=False,
        norm: Literal["rms", "layer"] = "rms",
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.do_selfattn = do_selfattn

        kwargs = dict(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mha_config=mha_config,
            mlp_ratio=mlp_ratio,
            ctxt_ratio=ctxt_ratio,
            simplify=simplify,
            norm=norm,
        )

        if do_selfattn:
            self.sa_layer_a = SADiTLayer(**kwargs)
            if bidirectional:
                self.sa_layer_b = SADiTLayer(**kwargs)
        self.ca_layer_a = CADiTLayer(**kwargs)
        if bidirectional:
            self.ca_layer_b = CADiTLayer(**kwargs)

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        if self.do_selfattn:
            self.sa_layer_a.set_backend(attn_type)
            if self.bidirectional:
                self.sa_layer_b.set_backend(attn_type)
        self.ca_layer_a.set_backend(attn_type)
        if self.bidirectional:
            self.ca_layer_b.set_backend(attn_type)
        return self.attn_type

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
        ctxt: Tensor,
        mask_a: Tensor | BlockMask | None = None,
        mask_b: Tensor | BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.do_selfattn:
            seq_a = self.sa_layer_a(seq=seq_a, ctxt=ctxt, mask=mask_a)
        seq_a = self.ca_layer_a(seq_q=seq_a, seq_k=seq_b, ctxt=ctxt, mask_k=mask_b)
        if self.bidirectional:
            if self.do_selfattn:
                seq_b = self.sa_layer_b(seq=seq_b, ctxt=ctxt, mask=mask_b)
            seq_b = self.ca_layer_b(seq_q=seq_b, seq_k=seq_a, ctxt=ctxt, mask_k=mask_a)
        return seq_a, seq_b

    def reset_parameters(self):
        if self.do_selfattn:
            self.sa_layer_a.reset_parameters()
            if self.bidirectional:
                self.sa_layer_b.reset_parameters()
        self.ca_layer_a.reset_parameters()
        if self.bidirectional:
            self.ca_layer_b.reset_parameters()


class DiTEncoder(nn.Module):
    """
    Stack of DiTBlocks.

    Parameters
    ----------
    hidden_dim : int
    num_heads : int
    num_layers : int
    mha_config : dict, optional
        Passed verbatim to each Attention layer.
    mlp_ratio : float
    ctxt_ratio : float
        Context dim = hidden_dim * ctxt_ratio (fed to Modulation).
    attn_type : str
        'torch', 'flex', 'flash', or 'flash-varlen'.
    do_selfattn : bool
        Prepend a self-attention layer to each cross-attention layer.
    bidirectional : bool
        seq_b also attends to seq_a (except in the last layer).
    simplify : bool
        Fuse MLP into the attention projection; skip separate MLP block.
    norm : 'rms' | 'layer'
        Normalisation type. Default: RMSNorm.
    intermediate_idx_to_return : int
        If >= 0, also return seq_a after this layer index.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mha_config: dict = None,
        mlp_ratio: float = 4.0,
        ctxt_ratio: float = 3.0,
        attn_type: str = "torch",
        do_selfattn: bool = False,
        bidirectional: bool = True,
        simplify: bool = False,
        norm: Literal["rms", "layer"] = "rms",
        intermediate_idx_to_return: int = -1,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.do_selfattn = do_selfattn
        self.bidirectional = bidirectional
        self.simplify = simplify

        self.return_intermediate = intermediate_idx_to_return >= 0
        self.intermediate_idx = intermediate_idx_to_return

        if mha_config is None:
            mha_config = {}
        mha_config["attn_type"] = attn_type
        self.layers = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mha_config=mha_config,
                    mlp_ratio=mlp_ratio,
                    ctxt_ratio=ctxt_ratio,
                    bidirectional=bidirectional and i != num_layers - 1,
                    do_selfattn=do_selfattn,
                    simplify=simplify,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )
        self.set_backend(attn_type)

    def set_backend(self, attn_type: str):
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.set_backend(self.attn_type)
        return self.attn_type

    def forward(
        self,
        seq_a: Tensor,
        seq_b: Tensor,
        ctxt: Tensor,
        mask_a: Tensor | BlockMask | None = None,
        mask_b: Tensor | BlockMask | None = None,
    ):
        if self.attn_type == "flex":
            assert seq_a.size(1) == seq_b.size(1), (
                "For flex attention, sequences must be of equal length."
            )
        intermediate_output = None
        for i, layer in enumerate(self.layers):
            seq_a, seq_b = layer(seq_a, seq_b, ctxt, mask_a, mask_b)
            if self.return_intermediate and i == self.intermediate_idx:
                intermediate_output = seq_a
        return seq_a, seq_b, intermediate_output

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
