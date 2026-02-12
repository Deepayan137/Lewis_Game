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

import re

def parse_descriptions(output, category=None):
    """
    Extract coarse and detailed descriptions from model output with fallback strategies.
    
    Args:
        output: Model generation output string
        category: Optional category name for better fallback parsing
    
    Returns:
        dict with 'thinking', 'coarse', 'detailed' keys
    """
    
    result = {
        "thinking": "",
        "coarse": "",
        "detailed": ""
    }
    
    # ========== Extract Thinking ==========
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()
    else:
        # Fallback: Extract thinking without closing tag
        thinking_fallback = re.search(r'<thinking>(.*?)(?=<coarse|<detailed|$)', output, re.DOTALL)
        if thinking_fallback:
            result["thinking"] = thinking_fallback.group(1).strip()
    
    # ========== Extract Coarse Description ==========
    # Strategy 1: Try with both tags
    coarse_match = re.search(r'<coarse>(.*?)</coarse>', output, re.DOTALL)
    
    if coarse_match:
        result["coarse"] = coarse_match.group(1).strip()
    else:
        # Strategy 2: Opening tag exists but no closing tag
        coarse_fallback = re.search(r'<coarse>(.*?)(?=<detailed|<thinking|$)', output, re.DOTALL)
        if coarse_fallback:
            content = coarse_fallback.group(1).strip()
            # Take first line or until newline
            result["coarse"] = content.split('\n')[0].strip()
        else:
            # Strategy 3: Look for "A photo of a" pattern
            photo_pattern = re.search(r'(A photo of a [^\n.]{5,50})', output, re.IGNORECASE)
            if photo_pattern:
                result["coarse"] = photo_pattern.group(1).strip()
    
    # ========== Extract Detailed Description ==========
    # Strategy 1: Try with both tags
    detailed_match = re.search(r'<detailed>(.*?)</detailed>', output, re.DOTALL)
    
    if detailed_match:
        result["detailed"] = detailed_match.group(1).strip()
    else:
        # Strategy 2: Opening tag exists but no closing tag
        detailed_fallback = re.search(r'<detailed>(.*?)(?=<coarse|<thinking|$)', output, re.DOTALL)
        if detailed_fallback:
            content = detailed_fallback.group(1).strip()
            # Take first sentence or until newline
            result["detailed"] = content.split('\n')[0].strip()
        else:
            # Strategy 3: Look for "The {category}" pattern
            if category:
                category_pattern = re.search(
                    rf'(The {re.escape(category)}[^\n]*?(?:\.|$))', 
                    output, 
                    re.IGNORECASE
                )
                if category_pattern:
                    result["detailed"] = category_pattern.group(1).strip()
            else:
                # Strategy 4: Look for any "The X" pattern
                the_pattern = re.search(r'(The [A-Za-z]+[^\n]{20,200}?\.)', output)
                if the_pattern:
                    result["detailed"] = the_pattern.group(1).strip()
    
    # ========== Cleanup ==========
    # Remove any remaining XML tags
    result["coarse"] = re.sub(r'</?[^>]+>', '', result["coarse"]).strip()
    result["detailed"] = re.sub(r'</?[^>]+>', '', result["detailed"]).strip()
    
    # Remove extra whitespace
    result["coarse"] = ' '.join(result["coarse"].split())
    result["detailed"] = ' '.join(result["detailed"].split())
    
    return result

# # Usage
# output = model.generate(...)
# descriptions = parse_descriptions(output)
# print(f"Coarse: {descriptions['coarse']}")
# print(f"Detailed: {descriptions['detailed']}")

def extract_reasoning_answer_term(text: str, term: str) -> str:
    """
    Extracts the value for a given term from the text.
    It first tries to match a quoted value, then an unquoted word.
    """
    # Step 1: Strip markdown code blocks if present
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    
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
    }
    pattern = patterns.get(term)
    if not pattern:
        return None
    match = re.search(pattern, text)
    if match:
        result = (match.group(1) or match.group(2)).strip()
        # Step 2: Strip extra single/double quotes
        result = result.strip("'\"")
        return result
    else:
        # Fallback if regex doesn't match.
        parts = text.split(term)
        if parts:
            result = re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
            return result
        return None