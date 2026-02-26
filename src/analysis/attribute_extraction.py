"""
Attribute extraction utilities for hallucination analysis.

Two extraction strategies:
  - Rule-based : for structured description text (fast, deterministic)
  - LLM-based  : for reasoning traces (handles two-image attribution problem)

LLM backend options:
  - 'local' : Qwen3B loaded via HuggingFace transformers
  - 'api'   : Qwen3-30B via Dashscope OpenAI-compatible API
"""

import re
import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Color Vocabulary
# ============================================================================

BASE_COLORS = [
    # Standard colors
    "red", "orange", "yellow", "green", "blue", "purple", "violet",
    "pink", "brown", "black", "white", "grey", "gray",
    # Extended / named colors
    "beige", "cream", "gold", "golden", "silver", "tan", "teal", "cyan",
    "magenta", "maroon", "navy", "olive", "coral", "salmon", "lavender",
    "ivory", "ebony", "charcoal", "crimson", "amber", "rust", "turquoise",
    "indigo", "lilac", "mint", "peach", "rose", "scarlet", "bronze",
    "copper", "chocolate", "caramel", "cinnamon", "khaki", "emerald",
    "ruby", "sapphire", "lime", "aqua", "fuchsia", "mauve",
]

COLOR_MODIFIERS = [
    "light", "dark", "bright", "pale", "deep", "rich", "vivid", "dull",
    "reddish", "bluish", "greenish", "yellowish", "brownish",
    "grayish", "greyish", "off",
]


# ============================================================================
# Rule-Based Extraction (for descriptions)
# ============================================================================

def extract_colors_rule_based(text: str) -> set:
    """
    Extract color attributes from text using a rule-based approach.

    Handles three patterns (in priority order):
      1. Hyphenated compounds  : reddish-brown, cream-colored
      2. Modifier + color      : light brown, dark blue
      3. Standalone color word : brown, blue

    Args:
        text: A description or attribute string.

    Returns:
        Set of color strings found in the text.
    """
    if not text:
        return set()

    text_lower = text.lower()
    found = set()

    # 1. Hyphenated compounds — include compound if either part is a known color
    for match in re.finditer(r'\b(\w+)-(\w+)\b', text_lower):
        compound = match.group(0)
        part1, part2 = match.group(1), match.group(2)
        if part1 in BASE_COLORS or part2 in BASE_COLORS:
            found.add(compound)

    # 2. Modifier + color  (e.g. "light brown")
    for modifier in COLOR_MODIFIERS:
        for color in BASE_COLORS:
            pattern = rf'\b{re.escape(modifier)}\s+{re.escape(color)}\b'
            if re.search(pattern, text_lower):
                found.add(f"{modifier} {color}")

    # 3. Standalone colors (also adds base components of compounds above)
    for color in BASE_COLORS:
        if re.search(rf'\b{re.escape(color)}\b', text_lower):
            found.add(color)

    return found


# ============================================================================
# Color Set Comparison Helpers
# ============================================================================

def colors_overlap(set_a: set, set_b: set) -> set:
    """
    Return colors from set_a that have a substring match in set_b.

    Substring matching handles compound colors:
      'brown' in set_b matches 'reddish-brown' in set_a.

    Args:
        set_a: Colors to check (e.g. ref description colors).
        set_b: Colors to match against (e.g. reasoning colors).

    Returns:
        Subset of set_a whose elements have a match in set_b.
    """
    matched = set()
    for ca in set_a:
        for cb in set_b:
            if ca in cb or cb in ca:
                matched.add(ca)
                break
    return matched


def colors_mismatch(ref_colors: set, query_colors: set) -> set:
    """
    Return colors from ref_colors that have NO match in query_colors.

    Used to identify attributes present in the reference description
    that are absent from the query — i.e. potential hallucination targets.

    Args:
        ref_colors: Colors extracted from the reference description.
        query_colors: Ground-truth colors from the query concept description.

    Returns:
        Subset of ref_colors with no counterpart in query_colors.
    """
    mismatched = set()
    for rc in ref_colors:
        has_match = any(rc in qc or qc in rc for qc in query_colors)
        if not has_match:
            mismatched.add(rc)
    return mismatched


# ============================================================================
# LLM-Based Extractor (for reasoning traces)
# ============================================================================

class AttributeExtractor:
    """
    LLM-based color attribute extractor for reasoning traces.

    Reasoning traces compare two images; we only want colors the model
    attributes to Image 1 (the query). A simple keyword search cannot
    resolve this two-image attribution problem reliably, hence the LLM.

    Backends:
      mode='local' — Qwen3B via HuggingFace transformers (thinking disabled)
      mode='api'   — Qwen3-30B via Dashscope API           (thinking disabled)
    """

    SYSTEM_PROMPT = (
        "You are a precise visual attribute extractor. "
        "Extract only the explicitly mentioned color attributes from reasoning text."
    )

    EXTRACTION_PROMPT_TEMPLATE = """\
You are analyzing a reasoning text produced during a visual comparison task.
A model compared two images: Image 1 (the query / first image) and Image 2 (the reference / second image).

Your task:
  Extract ONLY the color attributes the model explicitly mentions about Image 1 (the query / first image).
  Do NOT include colors mentioned about Image 2.
  Do NOT infer or guess colors. Only extract what is explicitly stated.

Return a valid JSON array of color strings.
Return [] if no colors about Image 1 are mentioned.

Examples:
  Input : "Image 1 shows a red ball, whereas Image 2 shows a blue cube."
  Output: ["red"]

  Input : "The first image depicts a white cat with brown spots on a green mat."
  Output: ["white", "brown"]

  Input : "The second image shows a pink toy. The first image appears different."
  Output: []

  Input : "Both images show similar-looking objects."
  Output: []

Reasoning text:
{reasoning_text}

Output (JSON array only, no explanation):"""

    def __init__(
        self,
        mode: str = "api",
        local_model_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            mode             : 'local' or 'api'.
            local_model_path : Path to Qwen3B weights  (required when mode='local').
            api_key          : Dashscope key; falls back to DASHSCOPE_API_KEY env var.
        """
        if mode not in ("local", "api"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'local' or 'api'.")

        self.mode = mode
        self.model = None
        self.tokenizer = None
        self.client = None

        if mode == "local":
            if not local_model_path:
                raise ValueError("local_model_path is required for mode='local'.")
            self._load_local_model(local_model_path)
        else:
            self._init_api_client(api_key)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _load_local_model(self, model_path: str):
        """Load Qwen3B via transformers with float16 and automatic device mapping."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading local model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Local model loaded successfully.")

    def _init_api_client(self, api_key: Optional[str]):
        """Initialise the OpenAI-compatible Dashscope client."""
        from openai import OpenAI

        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Pass api_key or set DASHSCOPE_API_KEY."
            )
        self.client = OpenAI(
            api_key=key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("Dashscope API client initialised.")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_messages(self, reasoning_text: str) -> list:
        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            reasoning_text=reasoning_text
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def _parse_llm_output(self, output: str) -> set:
        """
        Parse a JSON array from the model output.
        Falls back to rule-based extraction on parse failure.
        """
        output = output.strip()

        # Look for the first JSON array in the output
        match = re.search(r'\[.*?\]', output, re.DOTALL)
        if match:
            try:
                colors = json.loads(match.group(0))
                return {str(c).lower().strip() for c in colors if c}
            except json.JSONDecodeError:
                pass

        logger.warning(
            "Failed to parse LLM output as JSON; falling back to rule-based. "
            f"Output preview: {output[:200]!r}"
        )
        return extract_colors_rule_based(output)

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _local_inference(self, reasoning_text: str) -> set:
        """Qwen3B local inference with thinking mode disabled."""
        import torch

        messages = self._build_messages(reasoning_text)

        # enable_thinking=False suppresses the <think>...</think> block
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_llm_output(response)

    def _api_inference(self, reasoning_text: str) -> set:
        """Qwen3-30B API inference with thinking mode disabled."""
        messages = self._build_messages(reasoning_text)

        completion = self.client.chat.completions.create(
            model="qwen3-30b-a3b-instruct-2507",
            messages=messages,
            stream=False,
            max_tokens=128,
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )
        response = completion.choices[0].message.content or ""
        return self._parse_llm_output(response)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_colors_from_reasoning(self, reasoning_text: str) -> set:
        """
        Extract colors the model attributes to the query image (Image 1)
        from a reasoning trace that discusses two images.

        Args:
            reasoning_text: Model reasoning text from R2P or zero-shot entry.

        Returns:
            Set of color strings attributed to the query / first image.
        """
        if not reasoning_text or not reasoning_text.strip():
            return set()

        try:
            if self.mode == "local":
                return self._local_inference(reasoning_text)
            else:
                return self._api_inference(reasoning_text)
        except Exception as exc:
            logger.error(
                f"LLM extraction failed ({exc}); falling back to rule-based."
            )
            return extract_colors_rule_based(reasoning_text)

    def batch_extract(self, reasoning_texts: list) -> list:
        """
        Extract colors from a list of reasoning texts.

        Args:
            reasoning_texts: List of reasoning strings.

        Returns:
            List of color sets, one per input text.
        """
        results = []
        n = len(reasoning_texts)
        for i, text in enumerate(reasoning_texts):
            if i % 50 == 0:
                logger.info(f"  Extracting reasoning colors: {i}/{n}")
            results.append(self.extract_colors_from_reasoning(text))
        return results
