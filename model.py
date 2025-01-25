import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get batch size, sequence length, num heads, and head dimension
    batch_size, seq_len, num_heads, head_dim = xq.shape

    # Reshape the queries and keys to match the complex number format
    xq = xq.reshape(batch_size, seq_len, num_heads, -1, 2)
    xk = xk.reshape(batch_size, seq_len, xk.size(2), -1, 2)

    # Split real and imaginary parts
    xq_r, xq_i = xq.unbind(-1)
    xk_r, xk_i = xk.unbind(-1)

    # Expand freqs_cis for broadcasting
    # Original shape: [seq_len, dim/2, 2] -> [1, seq_len, 1, dim/2, 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    cos, sin = freqs_cis.unbind(-1)

    # Apply rotary embeddings
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    # Combine real and imaginary parts and reshape back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)

    xq_out = xq_out.reshape(batch_size, seq_len, num_heads, head_dim)
    xk_out = xk_out.reshape(batch_size, seq_len, xk.size(2), head_dim)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 576,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # Create the embedding matrix directly
        pos = torch.arange(self.max_position_embeddings).float()
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        emb = pos[:, None] * freqs[None, :]  # [max_pos, dim/2]
        # Shape: [max_pos, dim/2, 2] where last dim is (cos, sin)
        self.register_buffer(
            "freqs_cis", torch.stack([torch.cos(emb), torch.sin(emb)], dim=-1)
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis[:seq_len].to(q.device)  # [seq_len, dim/2, 2]
        return apply_rotary_emb(q, k, freqs_cis)


class LlamaSdpaAttention(nn.Module):
    def __init__(
        self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )

        # Apply rotary embeddings before repeating keys for GQA
        query_states, key_states = self.rotary_emb(query_states, key_states, seq_length)

        # Repeat keys and values for grouped-query attention
        if self.num_key_value_heads != self.num_heads:
            key_states = key_states.repeat_interleave(
                self.num_heads // self.num_key_value_heads, dim=2
            )
            value_states = value_states.repeat_interleave(
                self.num_heads // self.num_key_value_heads, dim=2
            )

        # Reshape for attention computation
        query_states = query_states.transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)
        key_states = key_states.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        value_states = value_states.transpose(
            1, 2
        )  # (batch, num_heads, seq_len, head_dim)

        # Convert attention mask to float and proper shape
        if attention_mask is not None:
            # Convert to float
            attention_mask = attention_mask.to(dtype=query_states.dtype)
            # Add head dimension and convert to additive mask
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                query_states.dtype
            ).min

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaSdpaAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights if configured
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self, "config") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss

        return logits
