

"""
GOTS-Faster selector.

Same greedy incremental-orthogonalization kernel as gots_fast_selector.py,
but the whole K-step greedy loop is captured into a CUDA graph (one graph per
(N, K, feature_dim, device) shape) and replayed with static input/output
buffers.

At batch size 1 the greedy loop launches ~10 small kernels per step, so its
runtime is dominated by Python and kernel-launch overhead rather than GPU
compute. Replaying a captured graph removes all of that overhead while
executing exactly the same kernels in exactly the same order, so the
selections are bitwise identical to the eager fallback kernel.

If graph capture is unavailable or fails (e.g. non-CUDA device), the selector
transparently falls back to the eager kernel.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger


eval_logger.info("[MMTok] gots_faster_selector.py loaded")


def gots_faster_kernel_impl(
    image_features: torch.Tensor,
    target_vision_tokens: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Optimized implicit-residual GOTS selection (eager kernel).

    This is the same arithmetic as gots_fast_selector._gots_fast_kernel_impl;
    it is written with CUDA-graph-capturable ops only (no GPU->CPU syncs), so
    the whole loop can be recorded once and replayed afterwards.

    Args:
        image_features: [N, feature_dim] tensor of vision features.
        target_vision_tokens: Number of vision tokens to select (K).
        eps: Numerical stability constant.

    Returns:
        selected_indices: [K] sorted LongTensor of selected vision token indices.
    """
    N = image_features.size(0)
    feature_dim = image_features.size(1)
    K = target_vision_tokens
    device = image_features.device

    if K <= 0 or N == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if K >= N:
        return torch.arange(N, dtype=torch.long, device=device)

    if image_features.dtype == torch.float32:
        X = image_features.detach()
        if not X.is_contiguous():
            X = X.contiguous()
    else:
        X = image_features.detach().to(dtype=torch.float32).contiguous()

    selected = torch.empty(K, dtype=torch.long, device=device)

    # basis[t] stores q_t: [K, d].
    basis = torch.empty((K, feature_dim), dtype=X.dtype, device=device)

    # coefficients[t, j] = <x_j, q_t>: [K, N].
    # Each row is contiguous and can be used as the output of torch.mv.
    coefficients = torch.empty((K, N), dtype=X.dtype, device=device)

    # Compute squared norms directly with a fused reduction to avoid the
    # sqrt-then-square round-trip of vector_norm(..., ord=2, dim=1).square_().
    residual_norm2 = torch.einsum("nd,nd->n", X, X)

    for step in range(K):
        # Keep pivot as a 1-D tensor: indexing with a 0-dim CUDA tensor
        # (e.g. X[pivot]) implicitly extracts the scalar on CPU, which is a
        # GPU->CPU sync and therefore illegal during CUDA graph capture.
        # index_select/index_fill_ with tensor indices are capture-safe and
        # perform exactly the same arithmetic.
        pivot = torch.argmax(residual_norm2).reshape(1)
        selected[step] = pivot

        # Use the corresponding row in the preallocated basis as q.
        q = basis[step]
        q.copy_(X.index_select(0, pivot).squeeze(0))

        if step > 0:
            # r_pivot = x_pivot - sum_l <x_pivot, q_l> q_l
            #
            # basis[:step].T: [d, step]
            # coefficients[:step, pivot]: [step]
            #
            # Fused into a single addmv: q <- q - basis[:step].T @ coefs.
            # This avoids a separate projection buffer and an extra sub_ kernel.
            torch.addmv(
                q,
                basis[:step].transpose(0, 1),
                coefficients[:step].index_select(1, pivot).squeeze(1),
                beta=1.0,
                alpha=-1.0,
                out=q,
            )

        # Normalize with the tracked residual energy.
        #
        # Do not use sqrt(e) + eps: the subsequent energy update assumes
        # q has unit norm. Clamp the squared norm before rsqrt instead.
        pivot_norm2 = residual_norm2.index_select(0, pivot).clamp_min(eps)
        q.mul_(torch.rsqrt(pivot_norm2))

        # coefficients[step, j] = <x_j, q>
        dot = coefficients[step]
        torch.mv(X, q, out=dot)

        # ||r_j_new||^2 = ||r_j_old||^2 - <x_j, q>^2
        residual_norm2.addcmul_(dot, dot, value=-1.0)
        residual_norm2.index_fill_(0, pivot, -torch.inf)

    # torch.sort is more JIT-friendly than tensor.sort().values.
    sorted_selected, _ = torch.sort(selected)
    return sorted_selected


class _GotsCudaGraph:
    """A captured CUDA graph of the greedy loop for one static shape."""

    def __init__(self, num_tokens: int, k: int, feature_dim: int, device: torch.device):
        self.static_input = torch.empty(
            (num_tokens, feature_dim), dtype=torch.float32, device=device
        )

        # Warm up on a side stream so that lazy initializations (cuBLAS
        # handles/workspaces, autotuned kernel choices, allocator blocks)
        # happen before capture.
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                gots_faster_kernel_impl(self.static_input, k)
        torch.cuda.current_stream(device).wait_stream(side_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = gots_faster_kernel_impl(self.static_input, k)

    def run(self, features_fp32: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(features_fp32)
        self.graph.replay()
        # Clone so later replays cannot overwrite a previously returned result.
        return self.static_output.clone()


# shape key -> _GotsCudaGraph, or False if capture failed for that shape.
_graph_cache: Dict[tuple, Union["_GotsCudaGraph", bool]] = {}
_MAX_CACHED_GRAPHS = 8


def gots_faster_select(
    image_features: torch.Tensor,
    target_vision_tokens: int,
) -> torch.Tensor:
    """
    GOTS selection with CUDA-graph dispatch and eager fallback.

    Returns:
        selected_indices: [K] sorted LongTensor of selected vision token indices.
    """
    N = image_features.size(0)
    feature_dim = image_features.size(1)
    K = target_vision_tokens
    device = image_features.device

    if K <= 0 or N == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if K >= N:
        return torch.arange(N, dtype=torch.long, device=device)

    if image_features.dtype == torch.float32:
        X = image_features.detach()
        if not X.is_contiguous():
            X = X.contiguous()
    else:
        X = image_features.detach().to(dtype=torch.float32).contiguous()

    if X.device.type != "cuda":
        return gots_faster_kernel_impl(X, K)

    key = (N, K, feature_dim, X.device.index)
    runner = _graph_cache.get(key)
    if runner is None:
        if len(_graph_cache) >= _MAX_CACHED_GRAPHS:
            # Drop all captured graphs; deleting them frees their memory pools.
            _graph_cache.clear()
        try:
            runner = _GotsCudaGraph(N, K, feature_dim, X.device)
        except Exception as capture_err:  # pragma: no cover
            eval_logger.warning(
                f"[MMTok] GOTS-Faster CUDA graph capture failed ({capture_err}); "
                "falling back to the eager kernel for this shape."
            )
            runner = False
        _graph_cache[key] = runner

    if runner is False:
        return gots_faster_kernel_impl(X, K)
    return runner.run(X)


class GOTSFasterCore:
    """CUDA-graph-accelerated GOTS core (batch size 1 oriented)."""

    def __init__(self, mmtok_core=None):
        self.mmtok_core = mmtok_core

    @torch.inference_mode()
    def select_vision_tokens(
        self,
        image_embeds: torch.Tensor,
        image_features: torch.Tensor,
        question: str = "",
        target_vision_tokens: int = None,
        image_grid_thw: torch.Tensor = None,
        text_token_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        del image_embeds, question, image_grid_thw, text_token_embedding
        return gots_faster_select(
            image_features=image_features,
            target_vision_tokens=target_vision_tokens,
        )

    @torch.inference_mode()
    def select_vision_tokens_batch(
        self,
        image_embeds_list,
        image_features_list,
        questions,
        target_vision_tokens_list,
        image_grid_thw_list=None,
    ):
        selected_indices_list = []
        num_samples = len(image_embeds_list)
        num_targets = len(target_vision_tokens_list)
        default_target = target_vision_tokens_list[-1]
        for i in range(num_samples):
            emb = image_embeds_list[i]
            feat = image_features_list[i]
            target = target_vision_tokens_list[i] if i < num_targets else default_target
            selected_indices = self.select_vision_tokens(
                image_embeds=emb,
                image_features=feat,
                target_vision_tokens=target,
            )
            selected_indices_list.append(selected_indices)
        return selected_indices_list


class GOTSFasterSelector:
    """MMTok-compatible wrapper for the CUDA-graph-accelerated GOTS kernel."""

    def __init__(
        self,
        target_vision_tokens: int = 32,
        alpha: float = 0.5,
    ):
        self.target_vision_tokens = target_vision_tokens
        self.alpha = alpha
        self._core = GOTSFasterCore(mmtok_core=None)

    @torch.inference_mode()
    def mm_coverage_selection(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_tokens_clip: torch.Tensor,
        tv_temp: float = 0.01,
        vv_temp: float = 0.2,
        padding_patch_indices: Optional[List[int]] = None,
        image_grid_thw: torch.Tensor = None,
    ) -> Tuple[List[int], torch.Tensor]:
        del text_token_embedding, tv_temp, vv_temp, padding_patch_indices, image_grid_thw

        selected_indices_tensor = self._core.select_vision_tokens(
            image_embeds=vision_tokens,
            image_features=vision_tokens_clip,
            target_vision_tokens=self.target_vision_tokens,
        )

        # Launch the GPU gather before .tolist(). Since .tolist() forces a
        # GPU-to-CPU synchronization, placing index_select first allows the
        # token gather to complete during the same synchronization.
        selected_tokens = torch.index_select(
            vision_tokens, dim=0, index=selected_indices_tensor
        )
        selected_indices = selected_indices_tensor.tolist()
        if not isinstance(selected_indices, list):
            selected_indices = [selected_indices]
        return selected_indices, selected_tokens
