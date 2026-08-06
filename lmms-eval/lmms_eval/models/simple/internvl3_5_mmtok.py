"""InternVL3.5 with MMTok/GOTS token selection for lmms-eval.

Follows the same retain_ratio pattern as Qwen2.5-VL MMTok since InternVL also
uses dynamic resolution (variable num_patches per image).
"""

import importlib
from typing import List, Tuple

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import (
    DEFAULT_GEN_KWARGS,
    InternVL3,
    load_image,
    load_video,
)
from mmtok.core import MMTokCore

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


@register_model("internvl3_5_mmtok")
class InternVL3_5_MMTok(InternVL3):
    """InternVL3.5 model wrapper with MMTok token selection.

    Args:
        pretrained: HuggingFace model path or local path.
        retain_ratio: Fraction of vision tokens to retain per image (default 0.1).
        selector_type: MMTok selector algorithm ("semantic" or "gots"; default "semantic").
        All other args are forwarded to InternVL3.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        retain_ratio: float = 0.1,
        selector_type: str = "semantic",
        **kwargs,
    ):
        # Store MMTok-specific config before calling parent init
        self._retain_ratio = retain_ratio
        self._selector_type = selector_type

        # Call InternVL3 init (loads model, tokenizer, accelerator, etc.)
        super().__init__(pretrained=pretrained, **kwargs)

        # MMTok requires a fast tokenizer for reliable is_split_into_words behaviour
        if not getattr(self.tokenizer, 'is_fast', False):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=True)
            eval_logger.warning(
                f"[MMTok-InternVL3.5] Replaced with fast tokenizer: {self._tokenizer.__class__.__name__}"
            )

        # Dynamically import get_conv_template from the remote-code package
        module_path = self.model.__module__.rsplit(".", 1)[0]
        conv_module = importlib.import_module(module_path + ".conversation")
        self._get_conv_template = conv_module.get_conv_template

        # Initialize MMTok core
        mmtok_kwargs = {
            "selector_type": self._selector_type,
            "device": self._device,
        }
        # Forward any extra MMTok params (alpha, temperatures, etc.) if present
        for key in ["alpha", "alpha_0", "softmax_tv_temperature", "softmax_vv_temperature", "target_vision_tokens"]:
            if key in kwargs:
                mmtok_kwargs[key] = kwargs[key]

        self._mmtok_core = MMTokCore(**mmtok_kwargs)
        self._mmtok_core.retain_ratio = self._retain_ratio
        self._mmtok_core._main_model_embed_tokens = self.model.language_model.get_input_embeddings()
        self._mmtok_core._language_tokenizer = self.tokenizer

        eval_logger.info(
            f"[MMTok-InternVL3.5] Injected MMTok: selector_type={self._selector_type}, "
            f"retain_ratio={self._retain_ratio}, device={self._device}"
        )

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses with per-image MMTok vision-token selection."""
        res: List[str] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # --- gen_kwargs sanitization (same as InternVL3) ---
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v
            pop_keys = [k for k, v in gen_kwargs.items() if k not in DEFAULT_GEN_KWARGS]
            for k in pop_keys:
                gen_kwargs.pop(k)

            # --- visual preprocessing (same as InternVL3) ---
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            if self.modality == "image":
                if visuals:
                    image_num = len(visuals)
                    dynamic_max_num = max(1, min(self.max_num, self.total_max_num // image_num))
                    processed_visuals = [
                        load_image(visual, max_num=dynamic_max_num).to(torch.bfloat16).to(self._device)
                        for visual in visuals
                    ]
                    pixel_values = torch.cat(processed_visuals, dim=0)
                    num_patches_list = [v.size(0) for v in processed_visuals]

                    existing_tags = contexts.count("<image>")
                    if existing_tags == 0:
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts
                    elif existing_tags == len(processed_visuals):
                        pass
                    else:
                        eval_logger.warning(
                            f"[InternVL3_5_MMTok] Token mismatch! Text has {existing_tags} tags but "
                            f"{len(processed_visuals)} images provided."
                        )
                        eval_logger.warning("[InternVL3_5_MMTok] Fallback: Prepending image tokens to the context.")
                        image_tokens = " ".join(["<image>"] * len(processed_visuals))
                        contexts = image_tokens + "\n" + contexts
                else:
                    pixel_values = None
                    num_patches_list = None

            elif self.modality == "video":
                assert len(visuals) == 1, f"Only one video is supported, but got {len(visuals)} videos."
                video_path = visuals[0]
                dynamic_max_num = max(1, min(self.max_num, self.total_max_num // self.num_frame))
                pixel_values, num_patches_list = load_video(
                    video_path, num_segments=self.num_frame, max_num=dynamic_max_num
                )
                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                contexts = video_prefix + contexts
            else:
                pixel_values = None
                num_patches_list = None

            # --- MMTok vision feature extraction + selection ---
            if pixel_values is not None:
                # Run vision tower
                if self.model.select_layer == -1:
                    vit_out = self.model.vision_model(
                        pixel_values=pixel_values,
                        output_hidden_states=False,
                        return_dict=True,
                    ).last_hidden_state
                else:
                    vit_out = self.model.vision_model(
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states[self.model.select_layer]
                vit_out = vit_out[:, 1:, :]  # drop CLS

                # Pixel shuffle + mlp1
                h = w = int(vit_out.shape[1] ** 0.5)
                vit_out = vit_out.reshape(vit_out.shape[0], h, w, -1)
                vit_out = self.model.pixel_shuffle(vit_out, scale_factor=self.model.downsample_ratio)
                vit_flat = vit_out.reshape(vit_out.shape[0], -1, vit_out.shape[-1])
                vit_projected = self.model.mlp1(vit_flat)  # [total_patches, num_image_token, llm_hidden]

                # Per-image MMTok selection
                start = 0
                selected_embeds_per_image = []
                for num_patches in num_patches_list:
                    end = start + num_patches
                    img_vit_projected = vit_projected[start:end]  # [num_patches, T, H]
                    img_vit_flat = vit_flat[start:end]            # [num_patches, T, H_vit]

                    img_vit_projected_flat = img_vit_projected.reshape(-1, img_vit_projected.shape[-1])
                    img_vit_flat_flat = img_vit_flat.reshape(-1, img_vit_flat.shape[-1])

                    target = int(self._mmtok_core.retain_ratio * img_vit_projected_flat.shape[0])
                    if target <= 0:
                        target = 1

                    _, selected_features = self._mmtok_core.apply_selection_preprocess_qwen(
                        image_embeds=img_vit_projected_flat,
                        image_features=img_vit_flat_flat,
                        question_text=contexts,
                        target_vision_tokens=target,
                    )
                    selected_embeds_per_image.append(selected_features)
                    start = end
            else:
                selected_embeds_per_image = []

            # --- Prompt construction (replicates chat() logic) ---
            question = contexts
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question

            template = self._get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep.strip())

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            if pixel_values is not None:
                for selected_features in selected_embeds_per_image:
                    num_selected_tokens = selected_features.shape[0]
                    image_tokens = (
                        IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * num_selected_tokens
                        + IMG_END_TOKEN
                    )
                    query = query.replace("<image>", image_tokens, 1)

            # Ensure img_context_token_id is set (normally done by chat())
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            self.model.img_context_token_id = img_context_token_id

            model_inputs = self.tokenizer(query, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(self._device)
            attention_mask = model_inputs["attention_mask"].to(self._device)
            gen_kwargs["eos_token_id"] = eos_token_id
            if "pad_token_id" not in gen_kwargs:
                gen_kwargs["pad_token_id"] = eos_token_id

            if pixel_values is not None:
                selected_vit_embeds = torch.cat(selected_embeds_per_image, dim=0)
                selected_vit_embeds = selected_vit_embeds.to(self._device).to(torch.bfloat16)
                generation_output = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual_features=selected_vit_embeds.unsqueeze(0),
                    **gen_kwargs,
                )
            else:
                generation_output = self.model.generate(
                    pixel_values=None,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep.strip())[0].strip()

            res.append(response)
            pbar.update(1)

        pbar.close()
        return res
