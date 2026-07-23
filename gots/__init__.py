

from .llava import mmtok, apply_llava_patches

# Patch llava.mm_utils.process_images at import time so that code which does
# "import mmtok" before "from llava.mm_utils import process_images" (or
# process_images = llava_mm_utils.process_images) gets the patched version.
# Padding indices are only computed when mmtok(model) is called with LLaVA-1.5.
apply_llava_patches()

__version__ = "1.0.0"
__all__ = ["mmtok", "apply_llava_patches"]
