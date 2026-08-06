# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# This file is modified from the official Qwen3-VL code (https://github.com/QwenLM/Qwen3-VL).
# Copyright (c) 2025 Zoom Communications, Inc. Author: Sixun Dong.

"""
MMTok Qwen3-VL model: overridden forward with token selection (image -> Vision Tower -> MMTok select -> Qwen3-VL).
"""

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger as eval_logger
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast, auto_docstring


class Qwen3_VL_MMTok(nn.Module):
    """
    MMTok Qwen3-VL: forward runs vision -> MMTok selection -> language model.
    """

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, **kwargs):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.
        Returns (image_embeds, image_features, deepstack_features) for MMTok.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        merged_hidden_states, last_hidden_state, deepstack_features = self.visual(
            pixel_values, grid_thw=image_grid_thw, **kwargs
        )
        merge_unit = self.visual.spatial_merge_size**2
        split_sizes = (image_grid_thw.prod(-1) // merge_unit).tolist()
        image_embeds = torch.split(merged_hidden_states, split_sizes)
        image_embeds = torch.cat(image_embeds, dim=0)

        # Pool the pre-merger last_hidden_state so that image_features has the same
        # number of tokens as image_embeds (one pooled feature per merged token).
        unmerged_split_sizes = (image_grid_thw.prod(-1)).tolist()
        last_hidden_state_split = torch.split(last_hidden_state, unmerged_split_sizes)
        pooled_features = []
        for ls in last_hidden_state_split:
            n_merged = ls.shape[0] // merge_unit
            # Truncate to a multiple of merge_unit (should already be exact)
            ls = ls[: n_merged * merge_unit]
            pooled_features.append(ls.view(n_merged, merge_unit, -1).mean(dim=1))
        image_features = torch.cat(pooled_features, dim=0)

        # deepstack features also need split + cat
        cat_deepstack = []
        if deepstack_features:
            for df in deepstack_features:
                split_df = torch.split(df, split_sizes)
                cat_deepstack.append(torch.cat(split_df, dim=0))

        return image_embeds, image_features, cat_deepstack

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None, **kwargs):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3VLModelOutputWithPast]:
        r"""
        MMTok patched forward for Qwen3-VL.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        select_pixel = False
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            select_pixel = True
            image_embeds, image_features, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, video_features, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Calculate RoPE index once per generation in the pre-fill stage only.
        # This runs on the FULL sequence (before token pruning); position_ids are
        # then filtered together with inputs_embeds / attention_mask below.
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None or (past_key_values is None or past_key_values.get_seq_length() == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # MMTok token selection for images
        keep_mask = None
        if select_pixel:
            question = self.get_question()
            token_retain_ratio = getattr(self._mmtok_core, "retain_ratio", float(os.environ.get("TOKEN_RETAIN_RATIO", "0.05")))
            target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])
            selection_method = os.environ.get("SELECTION_METHOD", "mmtok").lower()

            n = image_embeds.shape[0]
            if target_vision_tokens <= 0:
                selected_indices = torch.empty(0, dtype=torch.long, device=image_embeds.device)
                selected_image_embeds = image_embeds.new_empty(0, image_embeds.shape[-1])
            elif target_vision_tokens >= n:
                selected_indices = torch.arange(n, dtype=torch.long, device=image_embeds.device)
                selected_image_embeds = image_embeds[selected_indices]
            else:
                if selection_method == "mmtok":
                    selected_indices, selected_image_embeds = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds, image_features, question, target_vision_tokens=target_vision_tokens
                    )
                elif selection_method == "divprune":
                    def DivPrune(visual_feature_vectors, target_vision_tokens=1):
                        threshold_terms = target_vision_tokens

                        def pairwise_cosine_similarity(matrix):
                            norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
                            cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
                            return cosine_similarity

                        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

                        s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
                        selected_mask = torch.zeros(cosine_matrix.size(0), dtype=torch.bool, device=cosine_matrix.device)
                        for i in range(threshold_terms):
                            if i == 0:
                                m2 = cosine_matrix
                            else:
                                m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device)))

                            if i == 0:
                                scores = torch.topk(m2, 2, dim=0, largest=False).values[1, :]
                            else:
                                scores = torch.min(m2, dim=0).values

                            scores[selected_mask] = float("-inf")

                            phrase_to_add_idx = torch.argmax(scores)

                            s[i] = phrase_to_add_idx
                            selected_mask[phrase_to_add_idx] = True

                        s.sort()
                        return s

                    selected_indices = DivPrune(image_features, target_vision_tokens=target_vision_tokens)
                    selected_image_embeds = image_embeds[selected_indices]
                elif selection_method == "nahs_dls":
                    # NAHS-DLS algorithm implementation
                    X = image_embeds
                    k = target_vision_tokens
                    alpha_0 = 0.7

                    norms = torch.norm(X, p=2, dim=1)
                    candidate_mask = torch.ones(n, dtype=torch.bool, device=X.device)
                    selected = []

                    # Stage 1: Norm-guided initialization
                    first_idx = torch.argmax(norms).item()
                    selected.append(first_idx)
                    candidate_mask[first_idx] = False

                    if k > 1:
                        # Stage 2: Adaptive hybrid selection
                        for t in range(1, k):
                            alpha_t = alpha_0 * (1.0 - t / k)
                            beta_t = 1.0 - alpha_t

                            cand_X = X[candidate_mask]
                            sel_X = X[~candidate_mask]
                            cand_norms = norms[candidate_mask]
                            cand_indices = torch.where(candidate_mask)[0]

                            dists = torch.cdist(cand_X, sel_X, p=2)
                            min_dists = torch.min(dists, dim=1).values

                            norm_score = cand_norms / (torch.max(cand_norms) + 1e-8)
                            dist_score = min_dists / (torch.max(min_dists) + 1e-8)

                            scores = alpha_t * norm_score + beta_t * dist_score
                            sorted_idx = torch.argsort(scores, descending=True)
                            best_local = sorted_idx[0].item()
                            global_idx = cand_indices[best_local].item()

                            selected.append(global_idx)
                            candidate_mask[global_idx] = False

                    selected_indices = torch.tensor(selected, dtype=torch.long, device=image_embeds.device)
                    selected_image_embeds = image_embeds[selected_indices]
                else:
                    # Delegate any other selection method to MMTokCore
                    selected_indices, selected_image_embeds = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds, image_features, question, target_vision_tokens=target_vision_tokens
                    )

            if not isinstance(selected_indices, torch.Tensor):
                selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=image_embeds.device)

            select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
            select_mask[selected_indices] = True

            img_mask_1d = (input_ids == self.config.image_token_id)[0]
            st_idx = torch.nonzero(img_mask_1d, as_tuple=True)[0]

            if st_idx.numel() > 0:
                first, last = st_idx[0].item(), st_idx[-1].item()
                original_seq_len = input_ids.shape[1]
                keep_mask = torch.ones(original_seq_len, dtype=torch.bool, device=inputs_embeds.device)

                if len(selected_indices) == 0:
                    # Remove all vision tokens, keep text only
                    keep_mask[first : last + 1] = False
                    if position_ids is not None:
                        position_ids = torch.cat([position_ids[:, :, :first], position_ids[:, :, last + 1:]], dim=2)
                    if attention_mask is not None:
                        attention_mask = torch.cat([attention_mask[:, :first], attention_mask[:, last + 1:]], dim=1)
                    inputs_embeds = torch.cat([inputs_embeds[:, :first], inputs_embeds[:, last + 1:]], dim=1)
                    deepstack_image_embeds = None
                else:
                    # Keep only selected image tokens, remove unselected ones
                    keep_mask[first : last + 1] = False
                    selected_positions_local = first + selected_indices
                    keep_mask[selected_positions_local] = True

                    # Replace selected positions with selected embeds before filtering
                    selected_positions = first + selected_indices
                    inputs_embeds[:, selected_positions] = selected_image_embeds

                    # Filter sequence
                    if position_ids is not None:
                        position_ids = position_ids[:, :, keep_mask]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, keep_mask]
                    inputs_embeds = inputs_embeds[:, keep_mask]

                    # Filter deepstack features
                    if deepstack_image_embeds:
                        deepstack_image_embeds = [df[selected_indices] for df in deepstack_image_embeds]

                # Cache positions are local indices into the compacted sequence
                if cache_position is not None:
                    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

                del selected_indices, selected_image_embeds, select_mask, img_mask_1d, st_idx
                del question, token_retain_ratio, target_vision_tokens, selection_method

        # Rebuild visual_pos_masks and deepstack_visual_embeds for language model.
        visual_pos_masks = None
        deepstack_visual_embeds = None

        has_image = image_mask is not None
        has_video = video_mask is not None

        if has_image or has_video:
            new_seq_len = inputs_embeds.shape[1]
            visual_pos_masks = torch.zeros(1, new_seq_len, dtype=torch.bool, device=inputs_embeds.device)

            original_seq_len = input_ids.shape[1]
            if keep_mask is None:
                keep_mask = torch.ones(original_seq_len, dtype=torch.bool, device=inputs_embeds.device)

            old_kept_indices = torch.where(keep_mask)[0]
            old_to_new = torch.full((original_seq_len,), -1, dtype=torch.long, device=inputs_embeds.device)
            old_to_new[old_kept_indices] = torch.arange(len(old_kept_indices), device=inputs_embeds.device)

            # Image tokens after filtering
            if has_image and input_ids is not None:
                img_mask_1d = (input_ids == self.config.image_token_id)[0]
                if img_mask_1d.any():
                    image_indices = torch.where(img_mask_1d)[0]
                    kept_image_indices = image_indices[keep_mask[image_indices]]
                    if kept_image_indices.numel() > 0:
                        visual_pos_masks[:, old_to_new[kept_image_indices]] = True

            # Video tokens (always kept, but positions may have shifted)
            if has_video and input_ids is not None:
                vid_mask_1d = (input_ids == self.config.video_token_id)[0]
                if vid_mask_1d.any():
                    video_indices = torch.where(vid_mask_1d)[0]
                    kept_video_indices = video_indices[keep_mask[video_indices]]
                    if kept_video_indices.numel() > 0:
                        visual_pos_masks[:, old_to_new[kept_video_indices]] = True

            # Build deepstack_visual_embeds
            if has_image and has_video:
                # Mixed: need to merge image and video deepstack in visual_pos_masks order
                new_visual_indices = torch.where(visual_pos_masks[0])[0]
                if new_visual_indices.numel() > 0 and deepstack_image_embeds and deepstack_video_embeds:
                    img_mask_1d = (input_ids == self.config.image_token_id)[0]
                    vid_mask_1d = (input_ids == self.config.video_token_id)[0]
                    image_indices = torch.where(img_mask_1d)[0]
                    video_indices = torch.where(vid_mask_1d)[0]
                    kept_image_indices = image_indices[keep_mask[image_indices]]
                    kept_video_indices = video_indices[keep_mask[video_indices]]

                    image_new_positions = old_to_new[kept_image_indices]
                    video_new_positions = old_to_new[kept_video_indices]

                    image_in_visual = torch.isin(new_visual_indices, image_new_positions)
                    video_in_visual = torch.isin(new_visual_indices, video_new_positions)

                    deepstack_visual_embeds = []
                    for img_df, vid_df in zip(deepstack_image_embeds, deepstack_video_embeds):
                        embed_joint = torch.zeros(
                            len(new_visual_indices), img_df.shape[-1],
                            device=img_df.device, dtype=img_df.dtype
                        )
                        embed_joint[image_in_visual] = img_df
                        embed_joint[video_in_visual] = vid_df
                        deepstack_visual_embeds.append(embed_joint)
            elif has_image:
                if deepstack_image_embeds and len(deepstack_image_embeds) > 0:
                    deepstack_visual_embeds = deepstack_image_embeds
            elif has_video:
                if deepstack_video_embeds and len(deepstack_video_embeds) > 0:
                    deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            raise ValueError(
                "[MMTok] position_ids could not be computed. This forward expects either "
                "caller-provided position_ids or computable RoPE indices from input_ids."
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
