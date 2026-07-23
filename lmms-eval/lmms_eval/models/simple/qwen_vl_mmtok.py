"""Qwen-VL with MMTok (multiple selector support) for lmms-eval.

Follows the same retain_ratio/selector_type pattern as Qwen2.5-VL MMTok.
Qwen-VL represents each image with a fixed-size perceiver output; MMTok selects
a ``retain_ratio`` fraction of these vision tokens and rebuilds the input
embeddings directly so the language model sees only the selected tokens.
"""

import inspect
import os
import textwrap
import types
import uuid
from typing import List, Tuple

import torch
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers.cache_utils import Cache

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen_vl import Qwen_VL
from mmtok.core import MMTokCore


def _patch_qwen_vl_for_mmtok(model, device):
    """Patch Qwen-VL to support passing ``inputs_embeds`` to ``generate()``.

    Recent ``transformers`` versions pre-initialize ``past_key_values`` as a
    ``Cache`` object and initialize ``input_ids`` as an empty tensor when
    ``inputs_embeds`` is used. Qwen-VL's original code assumes (1) a truthy
    ``past_key_values`` always has tokens to slice, and (2) ``input_ids`` is
    always provided so it can detect image placeholders. These assumptions
    cause an ``IndexError`` / ``TypeError`` during generation with MMTok.
    """

    def patched_prepare_inputs_for_generation(
        input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        # Qwen-VL expects the legacy tuple cache format; drop new Cache objects.
        if isinstance(past_key_values, Cache):
            past_key_values = None

        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values and input_ids.shape[1] > 0:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values and input_ids.shape[1] > 0:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # When inputs_embeds is provided, transformers initializes input_ids as
        # empty. Use inputs_embeds on the first (prefill) step.
        if inputs_embeds is not None and input_ids.shape[1] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation

    # Patch the transformer forward by editing its source. The two changes are:
    #   1. Skip the in-forward image encoding when input_ids is None (MMTok has
    #      already built and injected the visual embeddings into inputs_embeds).
    #   2. Fix the legacy-cache past-length read: Qwen-VL stores cache tensors
    #      as [batch, seq_len, num_heads, head_dim], so seq_len is dim 1, not
    #      the second-to-last dimension.
    try:
        src = inspect.getsource(model.transformer.forward)
        src = textwrap.dedent(src)
        src = src.replace(
            "if past_key_values is None and torch.any(input_ids == self.config.visual['image_start_id']):",
            "if past_key_values is None and input_ids is not None and torch.any(input_ids == self.config.visual['image_start_id']):",
        )
        src = src.replace(
            "past_length = past_key_values[0][0].size(-2)",
            "past_length = past_key_values[0][0].size(1)",
        )
        module = inspect.getmodule(model.transformer.forward)
        namespace = {}
        exec(src, module.__dict__, namespace)
        model.transformer.forward = types.MethodType(namespace["forward"], model.transformer)
        eval_logger.info("[MMTok-Qwen-VL] Patched transformer.forward for inputs_embeds generation.")
    except Exception as exc:
        eval_logger.warning(
            f"[MMTok-Qwen-VL] Failed to source-patch transformer.forward ({exc}); "
            "falling back to torch.any wrapper."
        )
        original_transformer_forward = model.transformer.forward

        def patched_transformer_forward(self, *args, **kwargs):
            original_any = torch.any

            def safe_any(input, *args_any, **kwargs_any):
                if isinstance(input, bool):
                    return torch.tensor(False, dtype=torch.bool, device=device)
                return original_any(input, *args_any, **kwargs_any)

            torch.any = safe_any
            try:
                return original_transformer_forward(*args, **kwargs)
            finally:
                torch.any = original_any

        model.transformer.forward = types.MethodType(patched_transformer_forward, model.transformer)


@register_model("qwen_vl_mmtok")
class Qwen_VL_MMTok(Qwen_VL):
    """Qwen-VL model wrapper with MMTok token selection.

    Args:
        pretrained: HuggingFace model path or local path.
        retain_ratio: Fraction of vision tokens to retain per image (default 0.2).
        selector_type: MMTok selector algorithm (default "semantic").
        All other args are forwarded to Qwen_VL.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen-VL",
        retain_ratio: float = 0.2,
        selector_type: str = "semantic",
        **kwargs,
    ) -> None:
        # Store MMTok-specific config before calling parent init
        self._retain_ratio = retain_ratio
        self._selector_type = selector_type

        # Call Qwen_VL init (loads model, tokenizer, accelerator, etc.)
        super().__init__(pretrained=pretrained, **kwargs)

        # MMTok requires a fast tokenizer for reliable is_split_into_words behaviour
        if not getattr(self.tokenizer, "is_fast", False):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained, trust_remote_code=True, use_fast=True
            )
            self._tokenizer.padding_side = "left"
            self._tokenizer.pad_token_id = self._tokenizer.eod_id
            eval_logger.warning(
                f"[MMTok-Qwen-VL] Replaced with fast tokenizer: {self._tokenizer.__class__.__name__}"
            )

        # Initialize MMTok core
        mmtok_kwargs = {
            "selector_type": self._selector_type,
            "device": self._device,
        }
        for key in [
            "alpha",
            "alpha_0",
            "softmax_tv_temperature",
            "softmax_vv_temperature",
            "target_vision_tokens",
        ]:
            if key in kwargs:
                mmtok_kwargs[key] = kwargs[key]

        self._mmtok_core = MMTokCore(**mmtok_kwargs)
        self._mmtok_core.retain_ratio = self._retain_ratio
        self._mmtok_core._main_model_embed_tokens = self.model.transformer.get_input_embeddings()
        self._mmtok_core._language_tokenizer = self.tokenizer

        # Patch Qwen-VL so that generation accepts pre-built inputs_embeds.
        _patch_qwen_vl_for_mmtok(self.model, self._device)

        eval_logger.info(
            f"[MMTok-Qwen-VL] Injected MMTok: selector_type={self._selector_type}, "
            f"retain_ratio={self._retain_ratio}, device={self._device}"
        )

    @property
    def batch_size(self):
        # Per-image MMTok selection is easiest to reason about with batch_size=1
        return 1

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses with per-image MMTok vision-token selection."""
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)

        for chunk in re_ords.get_batched(n=1, batch_fn=None):
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            # Save images to /tmp because Qwen-VL visual encoder accepts paths
            visual_paths = []
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")

            # Set default until/max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            # Build query in Qwen-VL list format
            query = []
            if len(visual_paths) == 0:
                for context in contexts:
                    query.append({"text": context})
            else:
                for visual_path, context in zip(visual_paths, contexts):
                    query.append({"image": visual_path})
                    query.append({"text": context})

            questions = self.tokenizer.from_list_format(query)
            model_inputs = self.tokenizer(questions, return_tensors="pt", padding="longest")
            input_ids = model_inputs["input_ids"].to(self._device)
            attention_mask = model_inputs["attention_mask"].to(self._device)

            # Default generation kwargs
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except Exception:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eod_id

            if visual_paths:
                # Extract visual embeddings and apply MMTok per image
                visual_embeds = self.model.transformer.visual.encode(visual_paths)
                visual_embeds = visual_embeds.to(self._device)
                # visual_embeds: [num_images, num_visual_tokens, hidden_dim]

                question_text = " ".join(contexts).strip()
                selected_embeds_per_image = []
                for idx in range(visual_embeds.shape[0]):
                    single_embeds = visual_embeds[idx]  # [num_visual_tokens, hidden_dim]
                    target = int(self._mmtok_core.retain_ratio * single_embeds.shape[0])
                    if target <= 0:
                        target = 1

                    _, selected_features = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds=single_embeds,
                        image_features=single_embeds,  # Qwen-VL perceiver output used for both
                        question_text=question_text,
                        target_vision_tokens=target,
                    )
                    selected_embeds_per_image.append(selected_features)

                # Build inputs_embeds from token embeddings and insert selected visual tokens
                inputs_embeds = self.model.transformer.wte(input_ids)

                image_start_id = self.model.config.visual["image_start_id"]
                image_end_id = image_start_id + 1
                start_positions = (input_ids[0] == image_start_id).nonzero(as_tuple=True)[0]
                end_positions = (input_ids[0] == image_end_id).nonzero(as_tuple=True)[0]

                if start_positions.numel() != end_positions.numel():
                    eval_logger.warning("[MMTok-Qwen-VL] Mismatched image start/end tokens; skipping MMTok selection.")
                    inputs_embeds = inputs_embeds
                else:
                    # Process images from last to first so removals do not shift earlier positions
                    keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=self._device)
                    for a, b, selected_features in zip(
                        reversed(start_positions.tolist()),
                        reversed(end_positions.tolist()),
                        reversed(selected_embeds_per_image),
                    ):
                        target_len = selected_features.shape[0]
                        inputs_embeds[0, a + 1 : a + 1 + target_len] = selected_features.to(inputs_embeds.dtype)
                        # Drop old padding tokens between the selected visual tokens and </img>
                        keep_mask[a + 1 + target_len : b] = False
                    inputs_embeds = inputs_embeds[:, keep_mask]
                    attention_mask = attention_mask[:, keep_mask]

                cont = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eod_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                # When inputs_embeds is the prompt, generate() returns only new tokens.
                prompt_len = 0
            else:
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=self.tokenizer.eod_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                prompt_len = input_ids.shape[1]

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                cont_toks = cont_toks[prompt_len:]
                text_outputs = self.tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
                for term in until:
                    if len(term) > 0:
                        text_outputs = text_outputs.split(term)[0]

                res.append(text_outputs)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)

                for visual_path in visual_paths:
                    try:
                        os.remove(visual_path)
                    except Exception:
                        pass
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for requests. Not implemented for Qwen-VL MMTok."""
        raise NotImplementedError("Loglikelihood is not implemented for Qwen-VL MMTok.")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Generate multi-round responses. Not implemented for Qwen-VL MMTok."""
        raise NotImplementedError("Multi-round generation is not implemented for Qwen-VL MMTok.")
