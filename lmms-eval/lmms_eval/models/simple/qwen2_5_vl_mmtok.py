"""Qwen2.5-VL with MMTok/GOTS token selection for lmms-eval.

Thin wrapper around the base ``qwen2_5_vl`` model: loads the model exactly as
``Qwen2_5_VL`` does, then injects MMTok token selection via ``mmtok_qwen2_5_vl``.
"""

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from mmtok.qwen.qwen2_5_vl_mmtok import mmtok_qwen2_5_vl


@register_model("qwen2_5_vl_mmtok")
class Qwen2_5_VL_MMTok(Qwen2_5_VL):
    """Qwen2.5-VL with MMTok token selection.

    Args:
        pretrained: HuggingFace model path or local path.
        retain_ratio: Fraction of vision tokens to retain (default 0.2, i.e. keep 20%).
        selector_type: Token selector backend — "gots" or "semantic" (MMTok baseline).
        All other args are forwarded to Qwen2_5_VL.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        retain_ratio: float = 0.2,
        selector_type: str = "semantic",
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)

        # ========== Inject MMTok ==========
        self._model, self.processor = mmtok_qwen2_5_vl(
            self._model,
            language_tokenizer=self._tokenizer,
            processor=self.processor,
            retain_ratio=retain_ratio,
            selector_type=selector_type,
        )
        eval_logger.info(
            f"[MMTok-Qwen2.5] Wrapped qwen2_5_vl with MMTok: "
            f"retain_ratio={retain_ratio}, selector_type={selector_type}"
        )
