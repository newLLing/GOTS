# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
MMTok for Qwen-VL family: token selection injection and patching.
"""

from .qwen2_5_vl_mmtok import mmtok_qwen2_5_vl
from .qwen2_vl_mmtok import mmtok_qwen2_vl

__all__ = ["mmtok_qwen2_5_vl", "mmtok_qwen2_vl", "mmtok_qwen3_vl"]


def __getattr__(name):
    # Lazy import: Qwen3-VL requires transformers>=4.57, while the rest of this
    # package works with the pinned transformers==4.52.4. Importing it eagerly
    # would break Qwen2/2.5-VL users on the pinned version.
    if name == "mmtok_qwen3_vl":
        from .qwen3_vl_mmtok import mmtok_qwen3_vl

        return mmtok_qwen3_vl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
