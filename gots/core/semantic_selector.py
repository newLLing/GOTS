

"""
MMTok subset selection under maximum coverage.

Selects a subset of vision tokens to cover (i) text tokens (P: text–vision)
and (ii) the vision token set (Q: vision–vision diversity). Greedy selection
solves the maximum coverage formulation.
  - mm_coverage_selection: JIT kernel (greedy_merged_jit_kernel) for large n/k_max; optional padding via exclude_indices.
"""
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger as eval_logger


@torch.jit.script
def greedy_merged_jit_kernel(
    Combined: torch.Tensor,
    k_max: int,
    exclude_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Greedy maximum-coverage selection (JIT). Combined = [P; Q*alpha]:
    P = text–vision coverage, Q = vision–vision coverage (diversity).

    Args:
        Combined: [m+n, n] Float32, rows = text tokens + vision tokens, cols = vision tokens
        k_max: Number of vision tokens to select
        exclude_indices: 1D Long tensor of indices to exclude (e.g. padding); use empty tensor if none
    Returns:
        selected_indices: [k_max] Long, indices of selected vision tokens
    """
    total_rows, n = Combined.shape
    device = Combined.device
    dtype = Combined.dtype

    best_Combined = torch.zeros(total_rows, device=device, dtype=dtype)
    score_mask = torch.zeros(n, device=device, dtype=dtype)

    neg_inf_tensor = torch.tensor(float('-inf'), dtype=dtype, device=device)

    if exclude_indices.numel() > 0:
        score_mask.index_fill_(0, exclude_indices, neg_inf_tensor.item())

    selected_indices = torch.zeros(k_max, dtype=torch.long, device=device)

    for i in range(k_max):
        delta = (Combined - best_Combined.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        delta.add_(score_mask)
        best_idx = torch.argmax(delta)
        idx_1d = best_idx.view(1)
        selected_indices[i] = best_idx
    
        current_col = Combined.index_select(1, idx_1d).squeeze(1)
        score_mask.scatter_(0, idx_1d, neg_inf_tensor)
        best_Combined = torch.maximum(best_Combined, current_col)

    return selected_indices


class SemanticTokenSelector:
    """
    Coverage-based subset selection of vision tokens.

    Greedy selection to cover text tokens (P) and the vision token set (Q);
    score = coverage_gain(P) + alpha * coverage_gain(Q). mm_coverage_selection only.
    """

    def __init__(
        self,
        target_vision_tokens: int = 32,
        alpha: float = 0.5,
    ):
        self.target_vision_tokens = target_vision_tokens
        self.alpha: float = alpha

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    def mm_coverage_selection(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_tokens_clip: torch.Tensor,
            tv_temp: float = 0.01,
            vv_temp: float = 0.2,
        padding_patch_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Greedy maximum-coverage subset selection.
        Combined = [P; Q*alpha]: cover text tokens (P) and vision tokens (Q).
        Uses JIT kernel for large n/k_max; optional padding exclusion via exclude_indices.
        Returns (selected_indices, selected_vision_tokens).
        """
        device = vision_tokens.device
        x_norm = self._l2_normalize(vision_tokens).float()
        x_clip_norm = self._l2_normalize(vision_tokens_clip).float()
        z_norm = self._l2_normalize(text_token_embedding).float()  # question/keywords embedding

        P = z_norm @ x_norm.T   # text–vision coverage [m, n]
        Q = x_clip_norm @ x_clip_norm.T   # vision–vision coverage [n, n]
        m, n = P.shape
        inv_tv_temp = 1.0 / tv_temp
        inv_vv_temp = 1.0 / vv_temp
        P = F.softmax(P * inv_tv_temp, dim=1) / m
        Q = F.softmax(Q * inv_vv_temp, dim=1) / n
        k_max = min(self.target_vision_tokens, n)
        alpha = getattr(self, "alpha", 0.5)
        Q_weighted = Q * alpha
        Combined = torch.cat([P, Q_weighted], dim=0)   # [m+n, n] for maximum-coverage greedy
        
        n_threshold = int(os.getenv("MMTok_JIT_N_THRESHOLD", "500"))
        k_max_threshold = int(os.getenv("MMTok_JIT_K_MAX_THRESHOLD", "20"))
  
        use_jit = (n >= n_threshold) or (k_max >= k_max_threshold)
        if padding_patch_indices is not None:
            exclude_indices = torch.tensor(padding_patch_indices, device=device, dtype=torch.long)
        else:
            exclude_indices = torch.empty(0, dtype=torch.long, device=device)

        if use_jit:
            selected_indices = greedy_merged_jit_kernel(Combined, k_max, exclude_indices)
        else:
            best_Combined = torch.zeros(m + n, device=device, dtype=torch.float32)
            score_mask = torch.zeros(n, device=device, dtype=torch.float32)
            if padding_patch_indices is not None:
                score_mask[exclude_indices] = float("-inf")
            selected_indices = torch.empty(k_max, dtype=torch.long, device=device)
            neg_inf = float("-inf")
            for i in range(k_max):
                delta = (Combined - best_Combined.unsqueeze(1)).clamp_min_(0).sum(0)
                delta.add_(score_mask)
                best_idx = torch.argmax(delta)
                selected_indices[i] = best_idx
                torch.maximum(best_Combined, Combined[:, best_idx], out=best_Combined) 
                score_mask[best_idx] = neg_inf
        
        selected_indices, _ = selected_indices.sort()
        selected_tokens = vision_tokens[selected_indices]
        return selected_indices.tolist(), selected_tokens
