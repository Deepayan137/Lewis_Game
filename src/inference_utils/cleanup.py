import re

def extract_speaker_answer_term(text: str) -> str:
    """
    Return the content between <answer>...</answer> if present,
    otherwise fall back to everything after an opening <answer> or the original text.
    """
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # fallback: opening tag present but no closing tag
    fallback_pattern = r'<answer>\s*(.*?)\s*$'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        return fallback_match.group(1).strip()

    return text

def extract_reasoning_answer_term(text: str, term: str) -> str:
    """
    Extracts the value for a given term from the text.
    It first tries to match a quoted value, then an unquoted word.
    """
    patterns = {
        'Answer': r'"Answer":\s*(?:"([^"]+)"|([\w-]+))',
        'Confidence': r'"Confidence":\s*(?:"([^"]+)"|([\w.]+))',
        'Choice': r'"Choice":\s*(?:"([^"]+)"|([\w-]+))',
        'A': r'"A":\s*(?:"([^"]+)"|([\w-]+))',
        'B': r'"B":\s*(?:"([^"]+)"|([\w-]+))',
        'C': r'"C":\s*(?:"([^"]+)"|([\w-]+))',
        'D': r'"D":\s*(?:"([^"]+)"|([\w-]+))',
        'E': r'"E":\s*(?:"([^"]+)"|([\w-]+))',
        'F': r'"F":\s*(?:"([^"]+)"|([\w-]+))',
        # 'Caption': r'"Caption":\s*(?:"([^"]+)"|([\w-]+))'
    }
    pattern = patterns.get(term)
    if not pattern:
        return None
    match = re.search(pattern, text)
    if match:
        return (match.group(1) or match.group(2)).strip()
    else:
        # Fallback if regex doesn't match.
        parts = text.split(term)
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None