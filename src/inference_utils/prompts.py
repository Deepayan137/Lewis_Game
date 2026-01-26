"""
Prompt templates for different inference tasks.

This module centralizes all prompt generation for:
- Description generation (attribute-focused descriptions)
- Personalization (identify which reference matches query)
- Recognition (binary yes/no matching)
- VQA (visual question answering about personalized concepts)
"""

from typing import List, Optional


# ============================================================================
# Description Generation Prompts
# ============================================================================

def get_description_prompt(category: str) -> str:
    """
    Generate prompt for creating attribute-focused descriptions.

    Args:
        category: Object category (e.g., 'toy', 'pet animal', 'cup')

    Returns:
        Prompt string for description generation
    """
    return (
        f'Provide two descriptions of the {category} in the image:\n'
        f'1. A coarse 5-6 word description starting with "A photo of a "\n'
        f'2. A detailed description: Describe the {category} so it can be distinguished from other {category}s. '
        "Do NOT mention background, location or state. "
        "If the image contains a person, avoid mentioning clothing or accessories. "
        f'Write exactly one fluent sentence beginning with "The " and highlighting 3-4 visible distinguishing attributes. '
        "Keep it concise and natural, without lists or brackets.\n\n"
        "Output format:\n"
        "<thinking>Your reasoning</thinking>\n"
        f"<coarse>A photo of a ...</coarse>\n"
        f"<detailed>The ...</detailed>"
    )


# ============================================================================
# Personalization Prompts
# ============================================================================

def get_personalization_prompt(
    reference_info: str,
    names: List[str],
    num_options: int = 3,
) -> str:
    """
    Generate prompt for personalized identification task.

    Given a query image and multiple reference descriptions, identify which
    reference matches the query.

    Args:
        reference_info: Formatted string with reference descriptions
        names: List of option names (e.g., ['A', 'B', 'C'])
        num_options: Number of options

    Returns:
        Prompt string for personalization task
    """
    options_str = ", ".join(names[:num_options])

    prompt = (
        f"{reference_info}\n\n"
        "Task: Identify which reference matches the object in Image 1.\n\n"
        "Instructions:\n"
        "1. Compare the object in Image 1 with each reference description\n"
        "2. Focus on distinguishing visual attributes (color, shape, pattern, texture)\n"
        "3. Ignore background, lighting, and pose differences\n"
        "4. Select the reference that best matches\n\n"
        "Output format (JSON):\n"
        "{\n"
        '  "Reasoning": "Brief comparison of key attributes",\n'
        f'  "Answer": "<one of {options_str}>"\n'
        "}"
    )
    return prompt


def format_reference_info(
    descriptions: List[str],
    names: List[str],
) -> str:
    """
    Format reference descriptions for the prompt.

    Args:
        descriptions: List of description strings
        names: List of reference names (e.g., ['A', 'B', 'C'])

    Returns:
        Formatted reference info string
    """
    lines = ["Reference Descriptions:"]
    for name, desc in zip(names, descriptions):
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


# ============================================================================
# Recognition Prompts
# ============================================================================

def get_recognition_prompt(
    reference_info: str,
    question: Optional[str] = None,
    vqa: bool = False,
) -> str:
    """
    Generate prompt for binary recognition task.

    Given a query image (Image 1) and a reference image (Image 2) with its
    description, determine if they show the same object.

    Args:
        reference_info: Description of the reference object
        question: Optional custom question (for VQA mode)
        vqa: Whether this is a VQA task

    Returns:
        Prompt string for recognition task
    """
    if vqa and question:
        # VQA mode: answer a specific question
        prompt = (
            f"Reference Information:\n{reference_info}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Use the reference information to understand the personalized concept\n"
            "2. Answer the question based on Image 1\n"
            "3. Provide your reasoning before the answer\n\n"
            "Output format (JSON):\n"
            "{\n"
            '  "Reasoning": "Your analysis",\n'
            '  "Answer": "<A or B>"\n'
            "}"
        )
    else:
        # Binary recognition mode
        prompt = (
            f"Reference Description (Image 2):\n{reference_info}\n\n"
            "Task: Determine if Image 1 shows the same object as described for Image 2.\n\n"
            "Instructions:\n"
            "1. Compare the object in Image 1 with the reference description\n"
            "2. Focus on identity-defining attributes (not pose, background, or lighting)\n"
            "3. Answer 'Yes' if it's the same object, 'No' if different\n\n"
            "Output format (JSON):\n"
            "{\n"
            '  "Reasoning": "Brief comparison of key attributes",\n'
            '  "Answer": "<Yes or No>"\n'
            "}"
        )
    return prompt


# ============================================================================
# VQA Prompts
# ============================================================================

def get_vqa_prompt(
    reference_info: str,
    question: str,
    options: dict,
) -> str:
    """
    Generate prompt for visual question answering about personalized concepts.

    Args:
        reference_info: Description of the personalized concept
        question: The question to answer
        options: Dict of options (e.g., {'A': 'Yes', 'B': 'No'})

    Returns:
        Prompt string for VQA task
    """
    options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])

    prompt = (
        f"Reference Information:\n{reference_info}\n\n"
        f"Question: {question}\n"
        f"Options: {options_str}\n\n"
        "Instructions:\n"
        "1. Use the reference information to identify the personalized concept in Image 1\n"
        "2. Answer the question based on what you observe\n"
        "3. Select from the provided options\n\n"
        "Output format (JSON):\n"
        "{\n"
        '  "Reasoning": "Your analysis",\n'
        '  "Answer": "<option letter>"\n'
        "}"
    )
    return prompt


# ============================================================================
# Prompt Building Utilities
# ============================================================================

def build_concept_reference_block(
    concept_name: str,
    coarse_desc: str,
    detailed_desc: str,
) -> str:
    """
    Build a formatted reference block for a concept.

    Args:
        concept_name: Name/identifier of the concept
        coarse_desc: Short coarse description
        detailed_desc: Detailed description

    Returns:
        Formatted reference block string
    """
    return (
        f"<{concept_name}>:\n"
        f"  Coarse: {coarse_desc}\n"
        f"  Detailed: {detailed_desc}"
    )


def build_multi_reference_prompt(
    concepts: List[dict],
    task_instruction: str,
) -> str:
    """
    Build a prompt with multiple reference concepts.

    Args:
        concepts: List of dicts with 'name', 'coarse', 'detailed' keys
        task_instruction: The task instruction to append

    Returns:
        Complete prompt string
    """
    ref_blocks = []
    for concept in concepts:
        block = build_concept_reference_block(
            concept['name'],
            concept.get('coarse', ''),
            concept.get('detailed', ''),
        )
        ref_blocks.append(block)

    references = "\n\n".join(ref_blocks)
    return f"Reference Concepts:\n{references}\n\n{task_instruction}"
