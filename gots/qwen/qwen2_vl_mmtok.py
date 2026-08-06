# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# This file is modified from the official Qwen2-VL code (https://github.com/QwenLM/Qwen2-VL).
# Copyright (c) 2025 Zoom Communications, Inc. Author: Sixun Dong.

"""
MMTok for Qwen2-VL: injection and monkey-patching for token selection.
"""

import os
import types
from typing import Optional

import torch
from loguru import logger as eval_logger
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from ..core import MMTokCore
from .modeling_qwen2_vl_mmtok import Qwen2VisionTransformerPretrainedModel_MMTok


def mmtok_qwen2_vl(qwen_model, language_tokenizer=None, processor=None, retain_ratio=0.2, selector_type="semantic", **mmtok_kwargs):
    """
    Inject MMTok token selection into Qwen2-VL.

    Qwen2-VL uses dynamic resolution, so the number of vision tokens per image
    varies. MMTok uses a relative ``retain_ratio`` (fraction of tokens to keep)
    instead of an absolute count.

    Args:
        qwen_model: Qwen2-VL model instance
        language_tokenizer: Language tokenizer
        processor: Qwen2-VL processor (used to patch apply_chat_template for question hook)
        retain_ratio: Fraction of vision tokens to retain (default: 0.2, i.e. keep 20%).
        **mmtok_kwargs: Additional MMTok config (alpha, temperatures, etc.)

    Returns:
        (qwen_model, processor) with MMTok applied.

    Example:
        >>> from mmtok.qwen import mmtok_qwen2_vl
        >>> model, processor = mmtok_qwen2_vl(model, processor=processor, retain_ratio=0.2)
    """
    mmtok_config = {
        "alpha": 0.5,
        "softmax_tv_temperature": 0.01,
        "softmax_vv_temperature": 0.2,
        "device": qwen_model.device,
        "remove_padding_indices": False,  # only LLaVA 1.5 supports True; Qwen must be False
        "selector_type": selector_type,
        **mmtok_kwargs,
    }

    eval_logger.info(
        f"[MMTok-Qwen2] Injecting MMTok: retain_ratio={retain_ratio}, "
        f"device={mmtok_config['device']}"
    )
    mmtok_core = MMTokCore(**mmtok_config)
    mmtok_core.retain_ratio = retain_ratio
    eval_logger.info("[MMTok-Qwen2] MMTok core initialized")
    mmtok_core._main_model_embed_tokens = qwen_model.get_input_embeddings()
    mmtok_core._language_tokenizer = language_tokenizer
    qwen_model._mmtok_core = mmtok_core
    qwen_model._question_for_vision = None
    qwen_model.rope_deltas = None
    qwen_model.set_question = types.MethodType(_set_question, qwen_model)
    qwen_model.get_question = types.MethodType(_get_question, qwen_model)
    qwen_model.get_question_list = types.MethodType(_get_question_list, qwen_model)
    qwen_model.forward = types.MethodType(Qwen2_VL_MMTok.forward, qwen_model)
    qwen_model.model.visual.forward = types.MethodType(
        Qwen2VisionTransformerPretrainedModel_MMTok.forward, qwen_model.model.visual
    )
    eval_logger.info("[MMTok] Qwen2VLForConditionalGeneration.forward patched with MMTok")
    if processor is not None:
        patch_qwen2_vl_processor_for_question_hook(processor, qwen_model)
        eval_logger.info("[MMTok] Qwen2-VL processor.apply_chat_template patched for question hook")
    else:
        eval_logger.warning("[MMTok] No processor provided, skipping apply_chat_template patch")
    eval_logger.info("[MMTok-Qwen2] MMTok injection done")

    return qwen_model, processor


class Qwen2_VL_MMTok:
    """
    MMTok Qwen2-VL: forward runs vision -> MMTok selection -> language model.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        select_pixel = False
        image_embeds = None
        image_features = None

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                select_pixel = True
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds, image_features = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.get_dtype())
                video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                # visual now returns a tuple; take the first element as merged video embeds
                if isinstance(video_embeds, tuple):
                    video_embeds = video_embeds[0]
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # calculate RoPE index once per generation in the pre-fill stage only
        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if select_pixel:
            token_retain_ratio = getattr(self._mmtok_core, "retain_ratio", float(os.environ.get("TOKEN_RETAIN_RATIO", "0.1")))
            selection_method = os.environ.get("SELECTION_METHOD", "mmtok").lower()

            if selection_method == "divprune":
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
                            m2 = torch.index_select(
                                cosine_matrix, 0, torch.index_select(s, 0, torch.arange(0, i, device=cosine_matrix.device))
                            )

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
            else:
                DivPrune = None

            batch_size = input_ids.shape[0]
            if batch_size == 1:
                question = self.get_question()
                target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])

                if selection_method in ("mmtok", "gots"):
                    selected_indices, selected_image_embeds = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds, image_features, question, target_vision_tokens=target_vision_tokens, image_grid_thw=image_grid_thw
                    )
                elif selection_method == "divprune":
                    selected_indices = DivPrune(image_features, target_vision_tokens=target_vision_tokens)
                    selected_image_embeds = image_embeds[selected_indices]
                else:
                    raise ValueError(f"[MMTok] Unsupported SELECTION_METHOD: {selection_method}, use 'mmtok', 'gots', or 'divprune'")

                select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
                select_mask[selected_indices] = True

                img_mask = (input_ids == self.config.image_token_id)[0]
                st_idx = torch.nonzero(img_mask, as_tuple=True)[0]

                if st_idx.numel() > 0:
                    first, last = st_idx[0].item(), st_idx[-1].item()
                    if len(selected_indices) == 0:
                        position_ids = torch.cat([position_ids[:, :, :first], position_ids[:, :, last + 1:]], dim=2)
                        attention_mask = torch.cat([attention_mask[:, :first], attention_mask[:, last + 1:]], dim=1)
                        inputs_embeds = torch.cat([inputs_embeds[:, :first], inputs_embeds[:, last + 1:]], dim=1)
                        if mm_token_type_ids is not None:
                            mm_token_type_ids = torch.cat(
                                [mm_token_type_ids[:, :first], mm_token_type_ids[:, last + 1:]], dim=1
                            )
                    else:
                        img_mask[first : last + 1] = ~select_mask
                        img_mask = ~img_mask
                        selected_positions = first + torch.tensor(selected_indices, device=img_mask.device)
                        inputs_embeds[:, selected_positions] = selected_image_embeds

                        position_ids = position_ids[:, :, img_mask]
                        attention_mask = attention_mask[:, img_mask]
                        inputs_embeds = inputs_embeds[:, img_mask]
                        if mm_token_type_ids is not None:
                            mm_token_type_ids = mm_token_type_ids[:, img_mask]
                        del selected_positions
                # Cache positions are local indices into the compacted sequence
                if cache_position is not None:
                    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                del image_embeds, image_features, selected_indices, selected_image_embeds
                del select_mask, img_mask, st_idx
                del question, target_vision_tokens
            else:
                # Batched path: process each sample individually because selection is question-specific
                # and produces variable-length outputs that need per-sample compaction.
                questions = self.get_question_list() if hasattr(self, "get_question_list") else []
                if len(questions) < batch_size:
                    last_q = self.get_question() or ""
                    questions = questions + [last_q] * (batch_size - len(questions))

                spatial_merge_unit = getattr(self.model.visual, "spatial_merge_size", 2) ** 2
                tokens_per_image = []
                for g in image_grid_thw:
                    tokens_per_image.append(int(g[0].item() * g[1].item() * g[2].item()) // spatial_merge_unit)

                total_img_toks = sum(tokens_per_image)
                if total_img_toks != image_embeds.shape[0]:
                    eval_logger.warning(
                        f"[MMTok] Token count mismatch: expected {total_img_toks}, got {image_embeds.shape[0]}. "
                        f"Falling back to equal split."
                    )
                    n_images = len(tokens_per_image)
                    tokens_per_image = [image_embeds.shape[0] // n_images] * n_images

                image_tokens_per_sample = (input_ids == self.config.image_token_id).sum(dim=1).tolist()
                sample_image_indices = []
                img_idx = 0
                for b in range(batch_size):
                    n_toks = image_tokens_per_sample[b]
                    indices = []
                    cumsum = 0
                    while img_idx < len(tokens_per_image) and cumsum < n_toks:
                        indices.append(img_idx)
                        cumsum += tokens_per_image[img_idx]
                        img_idx += 1
                    sample_image_indices.append(indices)

                image_embeds_split = []
                image_features_split = []
                offset = 0
                for n_toks in tokens_per_image:
                    image_embeds_split.append(image_embeds[offset : offset + n_toks])
                    image_features_split.append(image_features[offset : offset + n_toks])
                    offset += n_toks

                new_inputs_embeds = []
                new_attention_masks = [] if attention_mask is not None else None
                new_position_ids = []

                for b in range(batch_size):
                    ids_b = input_ids[b : b + 1]
                    embeds_b = inputs_embeds[b : b + 1]
                    attn_b = attention_mask[b : b + 1] if attention_mask is not None else None
                    pos_b = position_ids[:, b : b + 1, :]

                    img_indices = sample_image_indices[b]
                    if len(img_indices) == 0:
                        new_inputs_embeds.append(embeds_b.squeeze(0))
                        if new_attention_masks is not None:
                            new_attention_masks.append(attn_b.squeeze(0))
                        new_position_ids.append(pos_b.squeeze(1))
                        continue

                    sample_image_embeds = torch.cat([image_embeds_split[i] for i in img_indices], dim=0)
                    sample_image_features = torch.cat([image_features_split[i] for i in img_indices], dim=0)

                    question = questions[b] if b < len(questions) else questions[-1]
                    target_vision_tokens = int(token_retain_ratio * sample_image_embeds.shape[0])

                    if selection_method in ("mmtok", "gots"):
                        sample_grid_thw = image_grid_thw[img_indices] if len(img_indices) > 0 else None
                        selected_indices, selected_image_embeds = self._mmtok_core.apply_selection_preprocess_qwen(
                            sample_image_embeds,
                            sample_image_features,
                            question,
                            target_vision_tokens=target_vision_tokens,
                            image_grid_thw=sample_grid_thw,
                        )
                    elif selection_method == "divprune":
                        selected_indices = DivPrune(sample_image_features, target_vision_tokens=target_vision_tokens)
                        selected_image_embeds = sample_image_embeds[selected_indices]
                    else:
                        raise ValueError(f"[MMTok] Unsupported SELECTION_METHOD: {selection_method}, use 'mmtok', 'gots', or 'divprune'")

                    select_mask = torch.zeros(sample_image_embeds.shape[0], dtype=torch.bool, device=sample_image_embeds.device)
                    select_mask[selected_indices] = True

                    img_mask = (ids_b == self.config.image_token_id)[0]
                    st_idx = torch.nonzero(img_mask, as_tuple=True)[0]

                    if st_idx.numel() > 0:
                        first, last = st_idx[0].item(), st_idx[-1].item()
                        if len(selected_indices) == 0:
                            pos_b = torch.cat([pos_b[:, :, :first], pos_b[:, :, last + 1:]], dim=2)
                            if attn_b is not None:
                                attn_b = torch.cat([attn_b[:, :first], attn_b[:, last + 1:]], dim=1)
                            embeds_b = torch.cat([embeds_b[:, :first], embeds_b[:, last + 1:]], dim=1)
                        else:
                            img_mask_local = img_mask.clone()
                            img_mask_local[first : last + 1] = ~select_mask
                            img_mask_local = ~img_mask_local
                            selected_positions = first + torch.tensor(selected_indices, device=img_mask_local.device)
                            embeds_b[:, selected_positions] = selected_image_embeds

                            pos_b = pos_b[:, :, img_mask_local]
                            if attn_b is not None:
                                attn_b = attn_b[:, img_mask_local]
                            embeds_b = embeds_b[:, img_mask_local]

                    new_inputs_embeds.append(embeds_b.squeeze(0))
                    if new_attention_masks is not None:
                        new_attention_masks.append(attn_b.squeeze(0))
                    new_position_ids.append(pos_b.squeeze(1))

                max_len = max(e.shape[0] for e in new_inputs_embeds)

                def _pad_2d(t, target_len):
                    pad_size = target_len - t.shape[0]
                    if pad_size <= 0:
                        return t
                    pad = torch.zeros(pad_size, t.shape[1], device=t.device, dtype=t.dtype)
                    return torch.cat([t, pad], dim=0)

                def _pad_1d(t, target_len):
                    pad_size = target_len - t.shape[0]
                    if pad_size <= 0:
                        return t
                    pad = torch.zeros(pad_size, device=t.device, dtype=t.dtype)
                    return torch.cat([t, pad], dim=0)

                def _pad_posids(t, target_len):
                    pad_size = target_len - t.shape[1]
                    if pad_size <= 0:
                        return t
                    pad = torch.zeros(t.shape[0], pad_size, device=t.device, dtype=t.dtype)
                    return torch.cat([t, pad], dim=1)

                inputs_embeds = torch.stack([_pad_2d(e, max_len) for e in new_inputs_embeds], dim=0)
                position_ids = torch.stack([_pad_posids(p, max_len) for p in new_position_ids], dim=1)
                if new_attention_masks is not None:
                    attention_mask = torch.stack([_pad_1d(a, max_len) for a in new_attention_masks], dim=0)

                del image_embeds, image_features
                del questions, tokens_per_image, sample_image_indices
                del image_embeds_split, image_features_split
                del new_inputs_embeds, new_position_ids
                if new_attention_masks is not None:
                    del new_attention_masks
            del token_retain_ratio, selection_method

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


def _set_question(self, question: str):
    """Set question on qwen_model; stored on model."""
    self._question_for_vision = question
    if not hasattr(self, '_question_for_vision_list'):
        self._question_for_vision_list = []
    self._question_for_vision_list.append(question)


def _get_question(self):
    """Get question from qwen_model."""
    return self._question_for_vision


def _get_question_list(self):
    """Get and clear the batched question list from qwen_model."""
    lst = getattr(self, '_question_for_vision_list', [])
    self._question_for_vision_list = []
    return lst


def patch_qwen2_vl_processor_for_question_hook(processor, mmtok_model_instance):
    """
    Patch processor.apply_chat_template to capture question text and set it on mmtok_model_instance.
    """
    original_apply_chat_template = processor.apply_chat_template

    def patched_apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        question_text = extract_question_from_messages(messages)
        if question_text:
            mmtok_model_instance.set_question(question_text)
        return original_apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )

    processor.apply_chat_template = patched_apply_chat_template


def extract_question_from_messages(messages):
    """
    Extract question text from Qwen2-VL message format.
    User messages may have content as str or list of {"type": "text", "text": "..."} / {"type": "image", ...}.
    """
    question_parts = []
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, str):
                question_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content = item.get("text", "")
                        if text_content:
                            question_parts.append(text_content)
    full_question = " ".join(question_parts).strip().replace("<image>", "").strip()
    return full_question if full_question else None


__all__ = ["mmtok_qwen2_vl", "extract_question_from_messages"]
