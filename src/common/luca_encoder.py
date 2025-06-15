#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/11/15 10:09
@project: LucaVirusTasks
@file: luca_encoder
@desc: xxxx
"""
import math
from typing import Optional, Tuple, Any, List, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
logger = logging.get_logger(__name__)


if is_flash_attn_2_available():
    # 加载FlashAttention
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
        print("Able to use FlashAttention2")
    except ImportError:
        print("Unable to use FlashAttention2.")
    print("#" * 50)
else:
    print("Unable to use FlashAttention2.")

try:
    from modeling_moe import MoE
except ImportError:
    from src.common.modeling_moe import MoE


class LucaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LucaFlashRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        from flash_attn.ops.rms_norm import rms_norm as __rms_norm
        self.flash_rms_norm = __rms_norm

    def forward(self, x):
        return self.flash_rms_norm(x, self.weight, self.eps)


class LucaFFN(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 activation_function: str = "gelu",
                 activation_dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = nn.functional.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        return x


class LucaSwiGLUFFN(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 multiple_of: int,
                 ffn_dim_multiplier: Optional[float],):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class LucaGroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = None, max_batch_size: int = 1, max_seq_len: int = 3072):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = ColumnParallelLinear(
            dim,
            n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wk = ColumnParallelLinear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wv = ColumnParallelLinear(
            dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wo = RowParallelLinear(
            n_heads * self.head_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            )

        self.cache_k = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                max_batch_size,
                max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
        self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv

        keys = self.cache_k[:bsz, : start_pos + seq_len]
        values = self.cache_v[:bsz, : start_pos + seq_len]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)


class LucaRotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码ROPE
    """
    def __init__(self, dim: int, theta: float = 10000.0, *_, **__):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: batch_size*num_heads, seq_len, head_dim
        head_dim = embed_dim
        :param x:
        :return:
        """
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(x, seq_dimension=-2)

        return self.apply_rotary_pos_emb(x, self._cos_cached, self._sin_cached)

    @classmethod
    def rotate_half(cls, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply_rotary_pos_emb(cls, x, cos, sin):
        cos = cos[:, : x.shape[-2], :]
        sin = sin[:, : x.shape[-2], :]

        return (x * cos) + (cls.rotate_half(x) * sin)


class LucaAttention(nn.Module):
    """
    常规的Attention
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            use_rotary_position_embeddings: bool = True,
            config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        self.use_rotary_position_embeddings = use_rotary_position_embeddings

        if self.use_rotary_position_embeddings:
            self.RoPE = LucaRotaryPositionEmbedding(self.head_dim, theta=config.rope_theta)

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        # print("LucaAttention Reset Parameters.")
        nn.init.xavier_uniform_(self.k_proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.out_proj.weight, gain=nn.init.calculate_gain("relu"))

        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # batch_size, seq_len, dim
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        if self.use_rotary_position_embeddings:
            query_states, key_states = self.RoPE(query_states), self.RoPE(key_states)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def _get_unpadding_data(attention_mask):
    """
    去掉padding
    :param attention_mask:
    :return:
    """
    seq_lens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seq_len_in_batch = seq_lens_in_batch.max().item()
    cu_seq_lens = F.pad(torch.cumsum(seq_lens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seq_lens,
        max_seq_len_in_batch,
    )


class LucaFlashAttention2(LucaAttention):
    """
    FlashAttention2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LucaFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError("LucaFlashAttention2 attention does not support output_attentions")
        attn_weights = None
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        # get query proj
        query_states = self._reshape(self.q_proj(hidden_states), -1, bsz)
        # get key, value proj
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0].transpose(1, 2)
            value_states = past_key_value[1].transpose(1, 2)
        elif is_cross_attention:
            # cross_attentions
            key_states = self._reshape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._reshape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
            value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        else:
            # self_attention
            key_states = self._reshape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._reshape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        """
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        """
        if self.use_rotary_position_embeddings:
            query_states, key_states = self.RoPE(query_states), self.RoPE(key_states)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=self.dropout
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _dtype_convect(self, v: torch.FloatTensor, convect_type: str):
        """
        数据类型转换
        :param v:
        :param convect_type:
        :return:
        """
        if convect_type == "low":
            try:
                v = v.to(dtype=torch.bfloat16, non_blocking=True)
            except Exception as e:
                print("QKV convect float32 to bfloat16 error.")
                print(e)
                v = v.to(dtype=torch.float16, non_blocking=True)
        else:
            try:
                v = v.to(dtype=torch.float32, non_blocking=True)
            except Exception as e:
                print("QKV convect to float32 error.")
                print(e)
        return v

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None
    ):
        # 转换数据类型
        if query_states.dtype != torch.float16 and query_states.dtype != torch.bfloat16:
            query_states = self._dtype_convect(query_states, convect_type="low")
            key_states = self._dtype_convect(key_states, convect_type="low")
            value_states = self._dtype_convect(value_states, convect_type="low")

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )
        # 转换数据类型
        if attn_output.dtype != torch.float32:
            attn_output = self._dtype_convect(query_states, convect_type="high")
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpadding_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LucaSdpaAttention(LucaAttention):
    """
    Sdqa Attention
    requirement torch 2.0.0 +
    """
    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions or layer_head_mask is not None:
            logger.warning(
                "LucaModel is using LucaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if (
                is_cross_attention
                and past_key_value is not None
                and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        if self.use_rotary_position_embeddings:
            query_states, key_states = self.RoPE(query_states), self.RoPE(key_states)

        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        # requirement torch 2.0.0 +
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


# 三种不同类型的Attention
Luca_ATTENTION_CLASSES = {
    "eager": LucaAttention,
    "sdpa": LucaSdpaAttention,
    "flash_attention_2": LucaFlashAttention2,
}


class LucaEncoderLayer(nn.Module):
    """
    Encoder Layer
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_dim = config.embed_dim if hasattr(config, "embed_dim") else config.hidden_size
        self.use_moe = config.use_moe if hasattr(config, "use_moe") else False
        self.use_swiglu_ffn = config.use_swiglu_ffn if hasattr(config, "use_swiglu_ffn") else False
        self.layer_norm_type = config.layer_norm_type
        if self.use_moe:
            print("LucaEncoderLayer use MoE!")
            assert hasattr(config, "moe_activation_function") and config.moe_activation_function in ACT2FN
            assert hasattr(config, "moe_num_experts") and config.moe_num_experts > 0
            assert hasattr(config, "moe_hidden_dim") and config.moe_hidden_dim > 0 and config.moe_hidden_dim > config.d_model
            assert hasattr(config, "moe_top_k") and config.moe_top_k > 0
            self.moe_activation_function = config.moe_activation_function
            self.moe_num_experts = config.moe_num_experts
            self.moe_hidden_dim = config.moe_hidden_dim
            self.moe_top_k = config.moe_top_k

        if self.layer_norm_type == "pre":
            # pre
            if config.layer_norm_name == "RMSNorm":
                self.pre_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.pre_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.pre_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        if config.attn_type == "gqa":
            self.self_attn = LucaGroupedQueryAttention(
                dim=self.embed_dim,
                n_heads=config.encoder_attention_heads,
                n_kv_heads=config.encoder_kv_attention_heads,
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len
            )
        else:
            self.self_attn = Luca_ATTENTION_CLASSES[config.attn_type](
                embed_dim=self.embed_dim,
                num_heads=config.encoder_attention_heads,
                dropout=config.attention_dropout,
                use_rotary_position_embeddings=config.use_rotary_position_embeddings,
                config=config,
            )
        if config.attn_type == "gqa":
            self.cross_attn = LucaGroupedQueryAttention(
                dim=self.embed_dim,
                n_heads=config.encoder_attention_heads,
                n_kv_heads=config.encoder_kv_attention_heads,
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len
            )
        else:
            self.cross_attn = Luca_ATTENTION_CLASSES[config.attn_type](
                self.embed_dim,
                config.encoder_attention_heads,
                dropout=config.attention_dropout,
                is_decoder=False,
                config=config,
            )

        if self.layer_norm_type != "pre":
            # post
            if config.layer_norm_name == "RMSNorm":
                self.self_attn_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.self_attn_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        else:
            # pre
            if config.layer_norm_name == "RMSNorm":
                self.post_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.post_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.post_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # cross
        if config.layer_norm_name == "RMSNorm":
            self.cross_attn_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
        elif config.layer_norm_name == "FlashRMSNorm":
            self.cross_attn_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
        else:
            self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            
        self.dropout = config.dropout
        if self.use_moe:
            self.moe = MoE(
                d_model=self.embed_dim,
                num_experts=self.moe_num_experts,
                hidden_dim=self.moe_hidden_dim,
                activation_function=self.moe_activation_function,
                top_k=self.moe_top_k
            )
        elif self.use_swiglu_ffn:
            self.swiglu_ffn = LucaSwiGLUFFN(
                dim=self.embed_dim,
                hidden_dim=4 * self.embed_dim,
                multiple_of=config.multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
            )
        else:
            self.ffn = LucaFFN(
                dim=self.embed_dim,
                hidden_dim=config.encoder_ffn_dim,
                activation_function=config.activation_function,
                activation_dropout=config.activation_dropout
            )

        # post
        if self.layer_norm_type != "pre":
            if config.layer_norm_name == "RMSNorm":
                self.final_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.final_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            cross_hidden_states: torch.Tensor = None,
            cross_attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True
    ) -> Tuple[Union[torch.FloatTensor, Any]]:
        residual = hidden_states
        if self.layer_norm_type == "pre":
            hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.layer_norm_type != "pre":
            hidden_states = self.self_attn_layer_norm(hidden_states)

            # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if cross_hidden_states is not None:
            residual = hidden_states
            if self.layer_norm_type == "pre":
                hidden_states = self.cross_attn_layer_norm(hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=cross_hidden_states,
                attention_mask=cross_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            if self.layer_norm_type != "pre":
                hidden_states = self.cross_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            if present_key_value is not None:
                present_key_value = present_key_value + cross_attn_present_key_value

        residual = hidden_states
        if self.layer_norm_type == "pre":
            hidden_states = self.post_layer_norm(hidden_states)
        if self.use_moe:
            # MoE FFN
            hidden_states = self.moe(hidden_states)
        elif self.use_swiglu_ffn:
            # SwiGLU FFN
            hidden_states = self.swiglu_ffn(hidden_states)
        else:
            # FFN
            hidden_states = self.ffn(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.layer_norm_type != "pre":
            hidden_states = self.final_layer_norm(hidden_states)

        if (hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16) and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LucaEncoder(PreTrainedModel):
    """
    Encoder
    """
    def __init__(self,
                 config: PretrainedConfig
                 ):
        super().__init__(config)
        self.dropout = config.dropout
        self.layer_dropout = config.encoder_layer_dropout
        self.use_rotary_position_embeddings = config.use_rotary_position_embeddings
        self.layer_norm_type = config.layer_norm_type

        self.embed_dim = config.embed_dim if hasattr(config, "embed_dim") else config.hidden_size
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        if self.layer_norm_type == "pre":
            self.embed_layer_norm = None
        else:
            if config.layer_norm_name == "RMSNorm":
                self.embed_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.embed_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.embed_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([LucaEncoderLayer(config) for _ in range(config.encoder_layers)])
        self._use_flash_attention_2 = config.attn_type == "flash_attention_2"
        self._use_sdpa = config.attn_type == "sdpa"
        if self.layer_norm_type == "pre":
            if config.layer_norm_name == "RMSNorm":
                self.last_layer_norm = LucaRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            elif config.layer_norm_name == "FlashRMSNorm":
                self.last_layer_norm = LucaFlashRMSNorm(dim=self.embed_dim, eps=config.layer_norm_eps)
            else:
                self.last_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_token_embeddings(self):
        return self.embed_tokens

    def set_token_embeddings(self, value):
        self.embed_tokens = value

    def get_token_type_embeddings(self):
        return self.embed_token_types

    def set_token_type_embeddings(self, value):
        self.embed_token_types = value

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            cross_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if hidden_states is not None:
            input_shape = hidden_states.size()[:-1]
        else:
            raise ValueError("You have to specify hidden_states")

        # expand attention_mask
        if attention_mask is not None:
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._use_sdpa and head_mask is None and not output_attentions:
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, hidden_states.dtype)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # expand cross attention mask
        if cross_hidden_states is not None and cross_attention_mask is not None:
            if self._use_flash_attention_2:
                cross_attention_mask = cross_attention_mask if 0 in cross_attention_mask else None
            elif self._use_sdpa and cross_attn_head_mask is None and not output_attentions:
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                cross_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    cross_attention_mask,
                    hidden_states.dtype,
                    tgt_len=input_shape[-1],
                )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                cross_attention_mask = _prepare_4d_attention_mask(
                    cross_attention_mask, hidden_states.dtype, tgt_len=input_shape[-1]
                )
                # hidden_states: batch_size, seq_len, embed_dim
        if self.layer_norm_type != "pre":
            hidden_states = self.embed_layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_encoder_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and cross_hidden_states is not None) else None
        next_encoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_encoder_states = all_encoder_states + (hidden_states,)
            # add layer_dropout (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training and self.layer_dropout > 0.0:
                dropout_probability = torch.rand([])
                # skip the layer
                if dropout_probability < self.layer_dropout:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None, None)
            else:
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        cross_hidden_states,
                        cross_attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                        None,
                        output_attentions,
                        use_cache
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        cross_hidden_states=cross_hidden_states,
                        cross_attention_mask=cross_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        cross_attn_layer_head_mask=(
                            cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                        ),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1], )
            if output_attentions and cross_hidden_states:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2], )

            if use_cache:
                next_encoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if self.layer_norm_type == "pre":
            hidden_states = self.last_layer_norm(hidden_states)

        if output_hidden_states:
            all_encoder_states = all_encoder_states + (hidden_states,)

        next_cache = next_encoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, all_encoder_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_encoder_states,
            past_key_values=next_cache,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

