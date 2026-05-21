# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Adapted for 3DGS generation (128x128 grid, 59 channels, 48 classes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """2D RoPE over flattened square patch tokens."""

    def __init__(self, dim, pt_seq_len, ft_seq_len=None, theta=10000):
        super().__init__()
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        freqs_1d = torch.einsum("n,d->nd", positions, freqs)
        freqs_1d = torch.repeat_interleave(freqs_1d, repeats=2, dim=-1)

        freqs_h = freqs_1d[:, None, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs_w = freqs_1d[None, :, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs_2d = torch.cat((freqs_h, freqs_w), dim=-1).reshape(ft_seq_len * ft_seq_len, -1)

        # Pre-shape (1, 1, N, D) and register every dtype variant up front.
        # The earlier `hasattr`-guarded lazy `.to()` cache forced a Dynamo
        # recompile on first forward and produced module-attribute Tensors that
        # got aliased across CUDA Graph replays under compile(reduce-overhead).
        # Buffers are non-persistent (derivable from ctor args).
        cos_4d = freqs_2d.cos().unsqueeze(0).unsqueeze(0)
        sin_4d = freqs_2d.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer("freqs_cos", cos_4d, persistent=False)
        self.register_buffer("freqs_sin", sin_4d, persistent=False)
        self.register_buffer("freqs_cos_bf16", cos_4d.bfloat16(), persistent=False)
        self.register_buffer("freqs_sin_bf16", sin_4d.bfloat16(), persistent=False)
        self.register_buffer("freqs_cos_fp16", cos_4d.half(), persistent=False)
        self.register_buffer("freqs_sin_fp16", sin_4d.half(), persistent=False)

    def forward(self, x):
        if x.dtype == torch.bfloat16:
            cos, sin = self.freqs_cos_bf16, self.freqs_sin_bf16
        elif x.dtype == torch.float16:
            cos, sin = self.freqs_cos_fp16, self.freqs_sin_fp16
        else:
            cos, sin = self.freqs_cos, self.freqs_sin
        return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight).to(input_dtype)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
    )


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.dropout(hidden))


class BottleneckPatchEmbed(nn.Module):
    """Patchify directly from the full-resolution grid via a low-rank bottleneck."""

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=59,
        bottleneck_dim=128,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        grid_size = img_size // patch_size
        self.num_patches = grid_size * grid_size

        self.proj1 = nn.Conv2d(
            in_chans,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(
            bottleneck_dim,
            embed_dim,
            kernel_size=1,
            stride=1,
            bias=bias,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(
                f"Input size ({h}x{w}) does not match model patch embed size "
                f"({self.img_size[0]}x{self.img_size[1]})."
            )
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    """Standard ViT-style patch embedding: single Conv2d from full grid → embed_dim tokens.

    Equivalent to a linear projection of each flattened patch (in_chans*patch_size^2 → embed_dim);
    no rank reduction below embed_dim, unlike BottleneckPatchEmbed.
    """

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=59,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        grid_size = img_size // patch_size
        self.num_patches = grid_size * grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(
                f"Input size ({h}x{w}) does not match model patch embed size "
                f"({self.img_size[0]}x{self.img_size[1]})."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Pre-compute the frequency vector once; register as non-persistent buffer
        # so it moves with the model (device-aware) without appearing in state_dict.
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("_freqs", freqs, persistent=False)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices (possibly fractional), one per batch element.
        :return: an (N, frequency_embedding_size) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        args = t[:, None].float() * self._freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEmbedder(nn.Module):
    """Projects the EOS-pooled (penultimate-normed) CLIP vector into hidden_size
    for the AdaLN pathway. On CFG drop, the caller-supplied ``drop_ids`` bool
    swaps the pooled vector for a cached null embedding (empty-string CLIP
    encoding) so the geometry matches conditional inputs.
    """

    def __init__(self, text_dim, hidden_size):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.register_buffer("null_emb", torch.zeros(text_dim), persistent=False)

    def forward(self, pooled, drop_ids=None):
        """pooled: (B, text_dim); drop_ids: (B,) bool or None (no drop)."""
        if drop_ids is not None:
            null = self.null_emb.to(pooled.dtype).expand_as(pooled)
            pooled = torch.where(drop_ids.unsqueeze(-1), null, pooled)
        return self.proj(pooled)


#################################################################################
#                                 Core JiT Model                                #
#################################################################################

class JiTBlock(nn.Module):
    """A JiT block: self-attn -> MLP, each gated by AdaLN-Zero.

    RMSNorm, qk-norm, RoPE on self-attn, SwiGLU MLP. Caption conditioning is
    injected through the AdaLN signal ``c`` (sum of time embedding and
    projected CLIP pool vector); there is no cross-attention pathway.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final JiT output projection."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """JiT-style diffusion backbone for 3DGS feature grids."""

    def __init__(
        self,
        input_size=128,
        patch_size=16,
        in_channels=59,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        text_dim=768,
        learn_sigma=False,
        gradient_checkpointing=True,
        bottleneck_dim=128,
        bottleneck=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch_size}")

        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.sample_size = input_size
        # JiT operates directly on the full 128x128 latent grid. Patchification is tokenization only,
        # not UNet/DiT-style spatial folding.
        self.spatial_fold_factor = 1
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"Per-head hidden size must be divisible by 4 for 2D RoPE, got hidden_size={hidden_size}, num_heads={num_heads}"
            )

        if bottleneck:
            self.x_embedder = BottleneckPatchEmbed(
                input_size,
                patch_size,
                in_channels,
                bottleneck_dim,
                hidden_size,
                bias=True,
            )
        else:
            self.x_embedder = PatchEmbed(
                input_size,
                patch_size,
                in_channels,
                hidden_size,
                bias=True,
            )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.class_dropout_prob = class_dropout_prob
        self.y_embedder = TextEmbedder(text_dim, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=self.head_dim // 2,
            pt_seq_len=input_size // patch_size,
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embed (single or two-stage) with xavier on the flattened weights.
        if isinstance(self.x_embedder, BottleneckPatchEmbed):
            w1 = self.x_embedder.proj1.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize text-cond projection MLP (standard xavier from _basic_init
        # already ran; null_emb stays at zero by design).

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in JiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module, rope):
        def ckpt_forward(x, c):
            return module(x, c, rope=rope)
        return ckpt_forward

    def load_null_embeddings(self, null_token):
        """Copy the cached empty-string CLIP pool vector into the null buffer.

        ``null_token``: (1, text_dim) or (text_dim,) tensor — the EOS-position
        token of the penultimate-normed empty-string CLIP encoding (i.e. the
        contents of `null_text_token.npz['null_token']`).
        """
        nt = null_token.detach().to(self.y_embedder.null_emb.dtype).reshape(-1)
        if nt.numel() != self.y_embedder.null_emb.numel():
            raise ValueError(
                f"null_token has {nt.numel()} elements, expected {self.y_embedder.null_emb.numel()}."
            )
        with torch.no_grad():
            self.y_embedder.null_emb.copy_(nt)

    def _draw_drop_ids(self, batch_size, device, force_drop_ids=None):
        """Decide which samples get the null conditioning this forward pass.

        Returns a (B,) bool tensor or None (no drop).
        """
        if force_drop_ids is not None:
            return force_drop_ids == 1
        if self.training and self.class_dropout_prob > 0:
            return torch.rand(batch_size, device=device) < self.class_dropout_prob
        return None

    def forward(self, x, t, y_pooled, force_drop_ids=None):
        """
        x:        (N, C, H, W)        — 3DGS feature grid (noisy)
        t:        (N,)                — diffusion timesteps
        y_pooled: (N, text_dim)       — EOS-pooled penultimate-normed CLIP vector
        force_drop_ids: optional (N,) 0/1 tensor; 1 = replace with null on this sample.
        """
        x = self.x_embedder(x) + self.pos_embed                                 # (N, T, D)
        t_emb = self.t_embedder(t)                                              # (N, D)
        drop_ids = self._draw_drop_ids(x.shape[0], x.device, force_drop_ids=force_drop_ids)
        y_pool = self.y_embedder(y_pooled, drop_ids=drop_ids)                   # (N, D)
        c = t_emb + y_pool                                                       # (N, D)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    self.ckpt_wrapper(block, self.feat_rope),
                    x, c,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, rope=self.feat_rope)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y_pooled, cfg_scale):
        """Batch conditional + unconditional halves for classifier-free guidance.

        The unconditional half is produced by setting ``force_drop_ids = 1``,
        which swaps the pooled CLIP vector for the cached null embedding.
        Caller passes an already-duplicated batch.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        N = combined.shape[0]
        force_drop = torch.cat(
            [torch.zeros(N // 2, device=x.device, dtype=torch.long),
             torch.ones(N // 2, device=x.device, dtype=torch.long)],
            dim=0,
        )
        model_out = self.forward(combined, t, y_pooled, force_drop_ids=force_drop)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                              3DGS DiT Configs                                 #
#################################################################################

def _jit_factory(*, depth, hidden_size, patch_size, num_heads, bottleneck_dim):
    return lambda **kw: DiT(
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
        bottleneck_dim=bottleneck_dim,
        **kw,
    )


JiT_3DGS_models = {
    'JiT-XL/8': _jit_factory(depth=28, hidden_size=1152, patch_size=8, num_heads=16, bottleneck_dim=256),
    'JiT-XL/16': _jit_factory(depth=28, hidden_size=1152, patch_size=16, num_heads=16, bottleneck_dim=256),
    'JiT-XL/32': _jit_factory(depth=28, hidden_size=1152, patch_size=32, num_heads=16, bottleneck_dim=256),
    'JiT-L/8': _jit_factory(depth=24, hidden_size=1024, patch_size=8, num_heads=16, bottleneck_dim=256),
    'JiT-L/16': _jit_factory(depth=24, hidden_size=1024, patch_size=16, num_heads=16, bottleneck_dim=256),
    'JiT-L/32': _jit_factory(depth=24, hidden_size=1024, patch_size=32, num_heads=16, bottleneck_dim=256),
    'JiT-B/8': _jit_factory(depth=12, hidden_size=768, patch_size=8, num_heads=12, bottleneck_dim=256),
    'JiT-B/16': _jit_factory(depth=12, hidden_size=768, patch_size=16, num_heads=12, bottleneck_dim=256),
    'JiT-B/32': _jit_factory(depth=12, hidden_size=768, patch_size=32, num_heads=12, bottleneck_dim=256),
    'JiT-S/8': _jit_factory(depth=12, hidden_size=384, patch_size=8, num_heads=6, bottleneck_dim=64),
    'JiT-S/16': _jit_factory(depth=12, hidden_size=384, patch_size=16, num_heads=6, bottleneck_dim=64),
    'JiT-S/32': _jit_factory(depth=12, hidden_size=384, patch_size=32, num_heads=6, bottleneck_dim=64),
}


DiT_3DGS_models = {
    **JiT_3DGS_models,
    'DiT-XL/8': _jit_factory(depth=28, hidden_size=1152, patch_size=8, num_heads=16, bottleneck_dim=256),
    'DiT-L/8': _jit_factory(depth=24, hidden_size=1024, patch_size=8, num_heads=16, bottleneck_dim=256),
    'DiT-B/8': _jit_factory(depth=12, hidden_size=768, patch_size=8, num_heads=12, bottleneck_dim=128),
    'DiT-S/8': _jit_factory(depth=12, hidden_size=384, patch_size=8, num_heads=6, bottleneck_dim=64),
}
