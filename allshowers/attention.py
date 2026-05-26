# Based on source code from: https://github.com/samvanstroud/hepattn
import torch
import torch.nn.functional as F
from torch import BoolTensor, Size, Tensor, nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
    flex_attention,
)
from torch.nn.functional import scaled_dot_product_attention

from allshowers.norms import RMSNorm

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None

ATTN_TYPES = {
    "torch": scaled_dot_product_attention,
    "flex": flex_attention,
    "flash": flash_attn_func,
    "flash-varlen": flash_attn_varlen_func,
}

# Which attention types support varlen / kv padding
VARLEN_ATTN_TYPES = [
    "torch",
    "flash-varlen",
]

# Which attention types support attention masking
ATTN_MASK_ATTN_TYPES = ["torch", "flex"]

# For now basically just defines which attention types expect (B, S, H, Dh) instead of (B, H, S, Dh)
FLASH_ATTN_TYPES = [
    "flash",
    "flash-varlen",
]


def create_padding_mask_from_kv(pads_kv: Tensor, safe: bool = True):
    if safe:
        S = pads_kv.size(1)

        def padding_safe(b, h, q_idx, kv_idx):
            is_valid_idx = kv_idx < S
            safe_kv_idx = torch.clamp(kv_idx, max=S - 1)
            return is_valid_idx & pads_kv[b, safe_kv_idx]

        return padding_safe

    def padding(b, h, q_idx, kv_idx):
        return pads_kv[b, kv_idx]

    return padding


def merge_masks(
    q_mask: BoolTensor | None,
    kv_mask: BoolTensor | None,
    attn_mask: BoolTensor | None,
    q_shape: Size,
    k_shape: Size,
) -> BoolTensor:
    """Create a full attention mask which incorporates the padding information.
    Modified from https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/salt/-/blob/main/salt/models/attention.py
    to use the convention that true slots are involved in computation / not masked out.
    """
    merged_mask = None

    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            merged_mask = kv_mask.unsqueeze(-2).expand(-1, q_shape[1], -1)
        elif kv_mask is None:
            merged_mask = q_mask.unsqueeze(-1).expand(-1, -1, k_shape[1])
        else:
            merged_mask = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2)

    if attn_mask is not None:
        merged_mask = attn_mask & merged_mask if merged_mask is not None else attn_mask

    return merged_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        value_residual: bool = False,
        qk_norm: bool = False,
        v_norm: bool = False,
        fuse_mlp: bool = False,
        mlp_ratio: float = 4.0,
        mlp_activation: str = "GELU",
        attn_gating: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "num_heads must divide dim."
        assert attn_type in ATTN_TYPES, f"Invalid attention type: {attn_type}"

        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = attn_type
        self.value_residual = value_residual
        self.qk_norm = qk_norm
        self.v_norm = v_norm
        self.fuse_mlp = fuse_mlp
        self.mlp_hidden_dim = int(dim * mlp_ratio) if fuse_mlp else 0
        self.mlp_activation = getattr(nn, mlp_activation)() if fuse_mlp else None
        self.attn_gating = attn_gating
        self.gating_size = self.num_heads if attn_gating else 0

        self.in_proj_weight = nn.Parameter(
            torch.empty(3 * dim + self.mlp_hidden_dim + self.gating_size, dim)
        )
        self.in_proj_bias = (
            nn.Parameter(torch.empty(3 * dim + self.mlp_hidden_dim + self.gating_size))
            if bias
            else None
        )
        self.out_proj = nn.Linear(dim + self.mlp_hidden_dim, dim, bias=bias)

        if self.value_residual:
            self.value_residual_mix = nn.Sequential(
                nn.Linear(dim, num_heads), nn.Sigmoid()
            )

        if self.qk_norm:
            self.q_norm = RMSNorm(dim)
            self.k_norm = RMSNorm(dim)
        if self.v_norm:
            self.v_norm_layer = RMSNorm(dim)

        self.reset_parameters()
        self.set_backend(attn_type, torch_compile=torch_compile)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.bias:
            nn.init.constant_(self.in_proj_bias, 0.0)
        self.out_proj.reset_parameters()

    def set_backend(self, attn_type: str, torch_compile: bool = False) -> str:
        if attn_type not in ATTN_TYPES:
            raise ValueError(f"Invalid attention type: {attn_type}")
        if attn_type in FLASH_ATTN_TYPES:
            if not _FLASH_AVAILABLE:
                print(
                    "Warning: flash_attn not installed, reverting to torch attention."
                )
                attn_type = "torch"
            elif (
                not torch.cuda.is_available()
                or torch.cuda.get_device_properties(0).major < 8
            ):
                print(
                    "Warning: Flash attention requires an NVIDIA GPU with compute capability >= 8.0, reverting to torch attention."
                )
                attn_type = "torch"

        self.attn_type = attn_type
        self.attn = ATTN_TYPES[attn_type]

        if torch_compile or attn_type == "flex":
            self.attn = torch.compile(self.attn)
        return self.attn_type

    def separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        x = x.unflatten(-1, (num_heads, -1))  # B S D -> B S H Dh
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B S H Dh -> B H S Dh
        return x

    def recombine_heads(self, x: Tensor) -> Tensor:
        if self.attn_type not in FLASH_ATTN_TYPES:
            x = x.transpose(-3, -2)  # B H S Dh -> B S H Dh
        return x.flatten(-2)  # B S H Dh -> B S D

    def _projection_packed(
        self, q: Tensor, kv: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if kv is None:
            qkv = F.linear(q, self.in_proj_weight, self.in_proj_bias)
            qkv, mlp, gating = torch.split(
                qkv,
                [3 * self.dim, self.mlp_hidden_dim, self.gating_size],
                dim=-1,
            )
            q, k, v = (
                qkv.unflatten(-1, (3, self.dim))
                .unsqueeze(0)
                .transpose(0, -2)
                .squeeze(-2)
                .contiguous()
            )
            return q, k, v, mlp, gating

        dim = q.size(-1)
        w_q, w_kv = self.in_proj_weight.split(
            [dim + self.mlp_hidden_dim + self.gating_size, dim * 2]
        )
        b_q, b_kv = (
            self.in_proj_bias.split(
                [dim + self.mlp_hidden_dim + self.gating_size, dim * 2]
            )
            if self.in_proj_bias is not None
            else (None, None)
        )

        q = F.linear(q, w_q, b_q)
        kv = F.linear(kv, w_kv, b_kv)
        k, v = (
            kv.unflatten(-1, (2, dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, mlp, gating = q.split(
            [self.dim, self.mlp_hidden_dim, self.gating_size], dim=-1
        )
        return q, k, v, mlp, gating

    def _prepare_qkv(
        self, q: Tensor, kv: Tensor | None = None, initial_values: dict | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mix = None
        if self.value_residual:
            mix = self.value_residual_mix(q)
            mix = mix.unsqueeze(-1)
            if self.attn_type not in FLASH_ATTN_TYPES:
                mix = mix.transpose(-2, -3)

        q, k, v, mlp, gating = self._projection_packed(q, kv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.v_norm:
            v = self.v_norm_layer(v)

        q = self.separate_heads(q, self.num_heads)
        k = self.separate_heads(k, self.num_heads)
        v = self.separate_heads(v, self.num_heads)

        if self.value_residual:
            if not initial_values:
                initial_values["v"] = v
            else:
                v = v * mix + initial_values["v"] * (1.0 - mix)
        return q, k, v, mlp, gating

    def _flash_varlen_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
    ) -> Tensor:
        bs, seqlen = q.shape[0], q.shape[1]
        if q_mask is None:
            q_mask = torch.ones((bs, q.shape[1]), dtype=torch.bool, device=q.device)
        if kv_mask is None:
            kv_mask = torch.ones((bs, k.shape[1]), dtype=torch.bool, device=k.device)
        q_flat, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q, q_mask.int())
        k_flat, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k, kv_mask.int())
        v_flat, _, _, _, _ = unpad_input(v, kv_mask.int())

        out = self.attn(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        )

        out = pad_input(out, indices_q, bs, seqlen=seqlen)
        return out

    def forward(
        self,
        q: Tensor,
        kv: Tensor | None = None,
        q_mask: BoolTensor | None = None,
        kv_mask: BoolTensor | None = None,
        attn_mask: BlockMask | BoolTensor | None = None,
        score_mod: _score_mod_signature | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        """
        Multi-head attention forward pass.

        Parameters
        ----------
        q : Tensor
            Queries tensor of shape (B, S, D).
        kv : Tensor, optional
            Keys/values tensor of shape (B, S, D). If None, defaults to self-attention.
        q_mask : BoolTensor, optional
            Query mask. True values are not padded.
        kv_mask : BoolTensor, optional
            Key/value mask. True values are not padded.
        attn_mask : BlockMask | BoolTensor, optional
            Attention mask. True values partake in computation.
        score_mod : _score_mod_signature, optional
            Score modifier for flex attention.
        initial_values : dict, optional
            Initial values for value residual connection.
        """
        if kv is None:
            q_shape = kv_shape = q.size()
        else:
            q_shape = q.size()
            kv_shape = kv.size()

        if kv_mask is not None:
            msg = f"Only the backends {VARLEN_ATTN_TYPES} support kv masking"
            assert self.attn_type in VARLEN_ATTN_TYPES, msg

        if attn_mask is not None:
            msg = f"Only the backends {ATTN_MASK_ATTN_TYPES} support attention masking"
            assert self.attn_type in ATTN_MASK_ATTN_TYPES, msg

        q, k, v, mlp, gating = self._prepare_qkv(q, kv, initial_values)
        if self.attn_type == "flash-varlen":
            out = self._flash_varlen_attention(q, k, v, q_mask=q_mask, kv_mask=kv_mask)
        elif self.attn_type == "flex":
            out = self.attn(q, k, v, block_mask=attn_mask, score_mod=score_mod)
        elif self.attn_type == "torch":
            attn_mask = merge_masks(q_mask, kv_mask, attn_mask, q_shape, kv_shape)
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(-3)
            out = self.attn(q, k, v, attn_mask=attn_mask)
        elif self.attn_type == "flash":
            out = self.attn(q, k, v)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        if self.attn_gating:
            gating = torch.sigmoid(gating)
            gating = gating.unsqueeze(-1)
            if self.attn_type not in FLASH_ATTN_TYPES:
                gating = gating.transpose(-2, -3)
            out = out * gating

        out = self.recombine_heads(out)
        if self.fuse_mlp:
            out = torch.cat([out, self.mlp_activation(mlp)], dim=-1)
        return self.out_proj(out)
