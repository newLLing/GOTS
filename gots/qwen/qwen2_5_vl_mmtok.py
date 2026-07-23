

"""
MMTok for Qwen2.5-VL: injection and monkey-patching for token selection.
"""

import os
import types

from loguru import logger as eval_logger

from ..core import MMTokCore
from .modeling_qwen2_5_vl_mmtok import Qwen2_5_VisionTransformerPretrainedModel_MMTok
from .qwen2_5_VLmodel_mmtok import Qwen2_5_VL_MMTok


def mmtok_qwen2_5_vl(qwen_model, language_tokenizer=None, processor=None, retain_ratio=0.2, **mmtok_kwargs):
    """
    Inject MMTok token selection into Qwen2.5-VL.

    Qwen2.5-VL uses dynamic resolution, so the number of vision tokens per image
    varies. MMTok uses a relative ``retain_ratio`` (fraction of tokens to keep)
    instead of an absolute count.

    Args:
        qwen_model: Qwen2.5-VL model instance
        language_tokenizer: Language tokenizer
        processor: Qwen2.5-VL processor (used to patch apply_chat_template for question hook)
        retain_ratio: Fraction of vision tokens to retain (default: 0.2, i.e. keep 20%).
        **mmtok_kwargs: Additional MMTok config (alpha, temperatures, selector_type, etc.).
            Set ``selector_type='gots'`` or ``SELECTION_METHOD=gots`` to use the
            CUDA-graph-accelerated GOTS selector instead of the default semantic
            coverage selector.

    Returns:
        (qwen_model, processor) with MMTok applied.

    Example:
        >>> from mmtok.qwen import mmtok_qwen2_5_vl
        >>> model, processor = mmtok_qwen2_5_vl(model, processor=processor, retain_ratio=0.2)
    """
    selection_method = os.environ.get("SELECTION_METHOD", "mmtok").lower()
    selector_type = "gots" if selection_method == "gots" else "semantic"

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
        f"[MMTok-Qwen2.5] Injecting MMTok: retain_ratio={retain_ratio}, "
        f"device={mmtok_config['device']}, selector_type={mmtok_config['selector_type']}"
    )
    mmtok_core = MMTokCore(**mmtok_config)
    mmtok_core.retain_ratio = retain_ratio
    eval_logger.info("[MMTok-Qwen2.5] MMTok core initialized")
    mmtok_core._main_model_embed_tokens = qwen_model.get_input_embeddings()
    mmtok_core._language_tokenizer = language_tokenizer
    qwen_model.model._mmtok_core = mmtok_core
    qwen_model.model._question_for_vision = None
    qwen_model.set_question = types.MethodType(_set_question, qwen_model)
    qwen_model.model.get_question = types.MethodType(_get_question, qwen_model.model)
    qwen_model.model.forward = types.MethodType(Qwen2_5_VL_MMTok.forward, qwen_model.model)
    qwen_model.model.get_video_features = types.MethodType(Qwen2_5_VL_MMTok.get_video_features, qwen_model.model)
    qwen_model.model.get_image_features = types.MethodType(Qwen2_5_VL_MMTok.get_image_features, qwen_model.model)
    qwen_model.model.visual.forward = types.MethodType(
        Qwen2_5_VisionTransformerPretrainedModel_MMTok.forward, qwen_model.model.visual
    )
    eval_logger.info("[MMTok] Qwen2_5_VLModel.forward patched with MMTok")
    if processor is not None:
        patch_qwen2_5_vl_processor_for_question_hook(processor, qwen_model)
        eval_logger.info("[MMTok] Qwen2.5-VL processor.apply_chat_template patched for question hook")
    else:
        eval_logger.warning("[MMTok] No processor provided, skipping apply_chat_template patch")
    eval_logger.info("[MMTok-Qwen2.5] MMTok injection done")

    return qwen_model, processor


def _set_question(self, question: str):
    """Set question on qwen_model; stored on model."""
    self.model._question_for_vision = question


def _get_question(self):
    """Get question from qwen_model.model."""
    return self._question_for_vision


def patch_qwen2_5_vl_processor_for_question_hook(processor, mmtok_model_instance):
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


def patch_qwen2_5_vl_for_question_hook(qwen_wrapper_instance, mmtok_model_instance):
    """
    Patch processor.apply_chat_template on the lmms_eval Qwen wrapper to capture question.
    """
    original_apply_chat_template = qwen_wrapper_instance.processor.apply_chat_template

    def patched_apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
        question_text = extract_question_from_messages(messages)
        if question_text:
            mmtok_model_instance.set_question(question_text)
        return original_apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )

    qwen_wrapper_instance.processor.apply_chat_template = patched_apply_chat_template


def extract_question_from_messages(messages):
    """
    Extract question text from Qwen2.5-VL message format.
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


__all__ = ["mmtok_qwen2_5_vl", "extract_question_from_messages"]
