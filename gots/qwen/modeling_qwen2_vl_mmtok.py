# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# This file is modified from the official Qwen2-VL code (https://github.com/QwenLM/Qwen2-VL).
# Copyright (c) 2025 Zoom Communications, Inc. Author: Sixun Dong.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel


class Qwen2VisionTransformerPretrainedModel_MMTok(Qwen2VisionTransformerPretrainedModel):
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `tuple`: (merged_hidden_states, img_features).
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        img_features = hidden_states
        hidden_states = self.merger(hidden_states)

        # Average spatial merge unit (2x2 = 4) for the pre-merger features
        img_features = img_features.reshape(img_features.shape[0] // 4, 4, -1).mean(dim=1)
        return hidden_states, img_features
