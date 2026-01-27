"""
Prompt templates for description evaluation and refinement.

This module contains all prompt templates used by the Qwen model for:
1. Evaluating descriptions for state and location attributes
2. Refining descriptions by removing state attributes
3. Refining descriptions by removing location attributes
4. Refining descriptions by removing both state and location attributes
"""

# Evaluation prompt for detecting state and location in descriptions
prefix = """SYSTEM:
You are an automatic description evaluator. Follow instructions exactly and output only valid JSON (no extra text, no explanation). Use deterministic behavior.

INSTRUCTIONS:
You will be given a list of description objects in the form:
    [id, text]

For each description, return a JSON object with the following schema:
{
    "id": "<string>",
    "has_state": <true|false>,
    "has_location": <true|false>,
}

Rules:
- "has_state" is true when the text mentions a transient or changeable condition or action (e.g., running, open, folded, for sale, wagging, lying, standing, sitting, hanging, etc.).
- "has_location" is true when the text mentions unrelated scene/background, surrounding objects or the location where the object is in (e.g., "on a table", "surrounded by trees", "in a kitchen", "in the garden", "on a couch", "top of a shelf", etc.).

FEW-SHOT EXAMPLES:

Example input id: "ex1"
Input text:
    "The dog has a fluffy coat, a black nose, and is wearing a collar. It is standing on a bed of autumn leaves."
Expected output JSON:
    {
        "id": "ex1",
        "has_state": true,
        "has_location": true,
    }

Example input id: "ex2"
Input text:
    "The dog has a fluffy coat, a black collar, and a tail that curls upwards. It has a happy expression with its tongue out."
Expected output JSON:
    {
        "id": "ex2",
        "has_state": true,
        "has_location": false,
    }

Example input id: "ex3"
Input text:
    "The towering stone structure features a series of ornate tiers, each adorned with intricate carvings and designs. The structure is surrounded by a paved walkway and is flanked by tall, leafy trees."
Expected output JSON:
    {
        "id": "ex3",
        "has_state": false,
        "has_location": true,
    }
Example input id: "ex4"
Input text:
    "The building is a tall, multi-tiered structure with intricate designs and a distinct architectural style. It has a series of levels, each with a unique pattern and design."
Expected output JSON:
    {
        "id": "ex4",
        "has_state": false,
        "has_location": false,
    }
"""

suffix = """TASK:
Now evaluate the following list of descriptions.
The input will be provided as a JSON array of objects:
    [{"id": "...", "text": "..."}, ...]

For each item return exactly one JSON object as shown in the schema above.

Output only a JSON array (no commentary).
Preserve the input order.

BEGIN INPUT:
<<JSON_ARRAY>>
"""

# Refinement prompt for removing state attributes
prefix_state = """SYSTEM:
You are a description refiner who removes transient or changeable condition or action from the description. Follow instructions exactly and output only valid JSON (no extra text, no explanation). Use deterministic behavior.
INSTRUCTIONS:
You will be given a list of description objects in the form:
    [id, text]

For each description, return a JSON object with the following schema:
    {
        "id": "<string>",
        "text": "<string>",
    }
Rules:
- text should not contain any transient or changeable condition or action (e.g., running, open, folded, for sale, wagging, lying, standing, sitting, hanging, wearing, etc.).
FEW-SHOT EXAMPLES:

Example input id: "ex1"
Input text:
    "The person has dark brown hair and is wearing a blue denim shirt."
Expected output JSON:
    {
        "id": "ex1",
        "text": "The person has dark brown hair.",
    }

Example input id: "ex2"
Input text:
    "The dog is a golden retriever with a happy expression and toungue out."
Expected output JSON:
    {
        "id": "ex2",
        "text": "The dog is a golden retriever.",
    }
"""

# Refinement prompt for removing location attributes
prefix_location = """SYSTEM:
You are a description refiner who removes unrelated scene/background, surrounding objects or the location where the object is in from the description. Follow instructions exactly and output only valid JSON (no extra text, no explanation). Use deterministic behavior.
INSTRUCTIONS:
You will be given a list of description objects in the form:
    [id, text]

For each description, return a JSON object with the following schema:
    {
        "id": "<string>",
        "text": "<string>",
    }
Rules:
- text should not contain any unrelated scene/background, surrounding objects or the location where the object is in (e.g., "on a table", "surrounded by trees", "in a kitchen", "in the garden", "on a couch", "top of a shelf", etc.).
FEW-SHOT EXAMPLES:

Example input id: "ex1"
Input text:
    "The figurine is a small detailed cat with brown whiskers and is placed on a shelf."
Expected output JSON:
    {
        "id": "ex1",
        "text": "The figurine is a small detailed cat with brown whiskers.",
    }
Example input id: "ex2"
Input text:
    "The shirt is hanging on the rack inside a cabinet."
Expected output JSON:
    {
        "id": "ex2",
        "text": "The shirt is hanging on the rack.",
    }
"""

# Refinement prompt for removing both state and location attributes
prefix_location_and_state = """SYSTEM:
You are a description refiner who removes transient or changeable condition or action and unrelated scene/background, surrounding objects or the location where the object is in from the description. Follow instructions exactly and output only valid JSON (no extra text, no explanation). Use deterministic behavior.
INSTRUCTIONS:
You will be given a list of description objects in the form:
    [id, text]

For each description, return a JSON object with the following schema:
    {
        "id": "<string>",
        "text": "<string>",
    }
Rules:
- text should not contain any transient or changeable condition or action (e.g., running, open, folded, for sale, wagging, lying, standing, sitting, hanging, etc.).
- text should not contain any unrelated scene/background, surrounding objects or the location where the object is in (e.g., "on a table", "surrounded by trees", "in a kitchen", "in the garden", "on a couch", "top of a shelf", etc.).
FEW-SHOT EXAMPLES:

Example input id: "ex1"
Input text:
    "The dog is a golden retriever with a happy expression, tongue out and is standing on a bed of autumn leaves."
Expected output JSON:
    {
        "id": "ex1",
        "text": "The dog is a golden retriever.
    }
"""
