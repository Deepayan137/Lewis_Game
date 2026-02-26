"""
Compute overlap between <detailed> (distinguishing features) and <state>/<location> fields.

Two modes:
  - Analysis metrics  (offline, requires corpus-level IDF):
      rouge_2     : raw bigram overlap F1
      cosine_sim  : TF-IDF cosine similarity
      rouge_l     : longest common subsequence F1  [kept for reference but unreliable — see notes]

  - Reward-compatible metric (no corpus dependency, safe to use inside GRPO reward):
      rouge_2_sw  : ROUGE-2 computed on stop-word-filtered content tokens only

Why rouge_2_sw for reward:
  TF-IDF cosine and ROUGE-L both require a pre-built corpus or are inflated by common
  words (see: `bo` false positive). Stripping stop words before bigram matching achieves
  the same noise suppression as TF-IDF weighting, with no corpus dependency — just a
  fixed hardcoded list. This makes it safe to call per-rollout during RL training.
"""

import json
import re
import math
from collections import Counter


# ---------------------------------------------------------------------------
# Stop word list  (hardcoded — no corpus needed)
# ---------------------------------------------------------------------------
# These are function words and generic verbs that carry no state/location signal.
# Content words that indicate state/location (lying, sitting, inside, background…)
# are intentionally kept so they can be matched against <state>/<location>.

STOP_WORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "both", "few", "more", "most", "other",
    "such", "same", "own",
    # pronouns
    "it", "its", "he", "she", "they", "them", "their", "there", "here",
    "i", "we", "you", "my", "our", "your", "his", "her",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "also", "either", "neither",
    # auxiliary / copula verbs  (keep action verbs like sitting/lying)
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    # common prepositions  (keep: inside, outside, near, behind — those carry location)
    "of", "to", "for", "with", "by", "as", "at", "from", "into", "onto",
    "upon", "about", "than", "through", "between", "among",
    # fillers / connectors
    "not", "no", "up", "out", "off", "then", "when", "where", "which",
    "who", "what", "how", "if", "while", "although", "because", "since",
    # generic appearance verbs (no state/location meaning)
    "appears", "seem", "seems", "features", "includes", "include",
    "makes", "give", "gives", "adds", "add",
    # numbers (generic)
    "one", "two", "three", "four", "five",
}


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_content(text: str) -> list[str]:
    """Tokenize and strip stop words. Used for reward-compatible ROUGE-2."""
    return [t for t in tokenize(text) if t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# ROUGE-N  (raw — no stop word filtering)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(hyp_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    hyp = _ngrams(hyp_tokens, n)
    ref = _ngrams(ref_tokens, n)
    overlap = sum(min(hyp[ng], ref[ng]) for ng in hyp)
    hyp_tot = sum(hyp.values())
    ref_tot = sum(ref.values())
    if hyp_tot == 0 or ref_tot == 0:
        return 0.0
    p = overlap / hyp_tot
    r = overlap / ref_tot
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

# ---------------------------------------------------------------------------
# ROUGE-2 with stop word filtering  — reward-compatible
# ---------------------------------------------------------------------------

def rouge_2_sw(hypothesis: str, reference: str) -> float:
    """
    ROUGE-2 F1 computed on content tokens only (stop words removed).

    Reward-compatible: requires only the two strings being compared at
    inference time — no corpus, no IDF table, no external model.
    Suppresses false positives from common words (a, the, with, and…)
    that inflated raw ROUGE-L scores.
    """
    return _rouge_n_f1(tokenize_content(hypothesis), tokenize_content(reference), 2)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_str(field) -> str:
    if isinstance(field, list):
        return " ".join(field)
    return str(field) if field else ""


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path  = "outputs/YoLLaVA/all/seed_23/descriptions_sp_ft_7b_detailed_analyze.json"
    output_path = "outputs/YoLLaVA/all/seed_23/overlap_analysis.json"

    with open(input_path) as f:
        data = json.load(f)

    # Build IDF from corpus (used only for cosine_sim analysis metric)
    corpus = []
    for concept in data.values():
        corpus.append(to_str(concept.get("distinguishing features", "")))
        corpus.append(to_str(concept.get("state", "")))
        corpus.append(to_str(concept.get("location", "")))

    results = {}

    for concept_name, concept in data.items():
        detailed_text = to_str(concept.get("distinguishing features", ""))
        state_text    = to_str(concept.get("state", ""))
        location_text = to_str(concept.get("location", ""))

        reference   = (state_text + " " + location_text).strip()
        sentences = split_sentences(detailed_text)
        sentence_scores = []

        for sent in sentences:
            r2_sw  = rouge_2_sw(sent, reference)
            sentence_scores.append({
                "sentence":   sent,
                "rouge_2_sw": round(r2_sw, 4)
            })

        sw_scores = [s["rouge_2_sw"] for s in sentence_scores]

        results[concept_name] = {
            "name":     concept_name,
            "category": concept.get("category", ""),
            "state":    state_text,
            "location": location_text,
            "sentence_scores": sentence_scores,
            "summary": {
                # rouge_2_sw is the primary metric for reward design
                "max_rouge_2_sw":  round(max(sw_scores),                   4) if sw_scores else 0.0,
                "mean_rouge_2_sw": round(sum(sw_scores) / len(sw_scores),  4) if sw_scores else 0.0,
            },
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_path}\n")

    # Console summary — sorted by rouge_2_sw descending
    print(f"{'Concept':<22} {'Category':<18} {'r2_sw':>6} {'mean':>6}  Worst sentence")
    print("-" * 105)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["summary"]["max_rouge_2_sw"]):
        s   = res["summary"]
        top = max(res["sentence_scores"], key=lambda x: x["rouge_2_sw"])
        flag = top["sentence"][:60] + "..." if top["rouge_2_sw"] > 0.1 else "-"
        print(f"{name:<22} {res['category']:<18} {s['max_rouge_2_sw']:>6.3f} {s['mean_rouge_2_sw']:>6.3f}  {flag}")


if __name__ == "__main__":
    main()
