

"""
VQA text processor: extract question/keywords for text token embedding (coverage over text).
"""
import re
from typing import Optional

import torch
from loguru import logger as eval_logger


# Stopwords with little visual relevance; keep descriptive words for text–vision coverage
NON_VISUAL_WORDS = {
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "it", "they", "he", "she", "we", "you", "i",
    "what", "where", "when", "why", "how", "who", "which",
    "and", "or", "but", "so", "if", "then",
    "question", "answer",
    "to", "for", "with", "by", "from", "at", "into", "onto", "upon",
}

WORD_REGEX = re.compile(r"\b\w+\b")
PROMPT_1 = "Answer the question using a single word or phrase."
PROMPT_2 = "Answer with the option's letter from the given choices directly."


class VQATextProcessor:
    """
    Text processor for MMTok: extract question/keywords used as text tokens
    in the coverage criterion (cover text tokens + vision token set).
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        eval_logger.info("[TextProcessor] Initialized (simple keyword extraction for coverage)")

    def extract_keywords_simple(self, text: str, max_words: Optional[int] = None) -> str:
        """
        Filter stopwords and return descriptive text for text token embedding (coverage).
        """
        if not text:
            return ""
        text = text.replace(PROMPT_1, "").replace(PROMPT_2, "")
        words = WORD_REGEX.findall(text)
        filtered_words = [w for w in words if w.lower() not in NON_VISUAL_WORDS]
        filtered_str = " ".join(filtered_words) if filtered_words else text
        return filtered_str
