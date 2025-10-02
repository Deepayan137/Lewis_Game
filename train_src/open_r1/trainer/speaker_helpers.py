import os
import json
import time
import requests, traceback
from urllib.parse import urlparse
from typing import List, Any, Optional, Dict
from datetime import datetime
import sys
import string

sys.path.insert(0, 'src/virft/src/')
from open_r1.dist_helpers import *



SPEAKER_URL = os.environ.get("SPEAKER_URL", "http://as03r3b07:9000/batch_describe")
SPEAKER_TIMEOUT = float(os.environ.get("SPEAKER_TIMEOUT", 30.0))  # seconds
_speaker_cache = {}
# Defaults you can tune via environment variables:
DEFAULT_SPEAKER_CHUNK_SIZE = int(os.environ.get("SPEAKER_CHUNK_SIZE", "4"))
DEFAULT_CHUNK_DELAY = float(os.environ.get("SPEAKER_CHUNK_DELAY", "0.05"))  # seconds between chunks
DEFAULT_MAX_RETRIES = int(os.environ.get("SPEAKER_MAX_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.environ.get("SPEAKER_BACKOFF_FACTOR", "1.0"))

def _call_speaker_batch(batch_requests, timeout=SPEAKER_TIMEOUT,
                         max_retries=DEFAULT_MAX_RETRIES,
                         backoff_factor=DEFAULT_BACKOFF,
                         chunk_size=DEFAULT_SPEAKER_CHUNK_SIZE,
                         chunk_delay=DEFAULT_CHUNK_DELAY):
    """
    Robust client-side call to speaker with chunking and retries.

    batch_requests: list of dicts: {"candidate_paths": [...], "question": "..."}
    Returns: list of responses aligned with batch_requests (each response is dict).
    """
    if not batch_requests:
        return []

    url = os.environ.get("SPEAKER_URL", "http://as03r3b07:9000/batch_describe")
    parsed = urlparse(url)
    host = parsed.hostname or ""

    # Decide whether to honor environment proxies. Default: do NOT trust env proxies
    # unless the host is explicitly present in NO_PROXY/no_proxy.
    no_proxy_env = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    no_proxy_hosts = [h.strip() for h in no_proxy_env.split(",") if h.strip()]
    use_trust_env = host in no_proxy_hosts

    sess = requests.Session()
    sess.trust_env = bool(use_trust_env)

    results_all = []
    total = len(batch_requests)
    connect_timeout = min(20.0, timeout)
    read_timeout = timeout

    # helper to build neutral fallback responses for a slice
    def neutral_resp_slice(slice_requests):
        return [{"predicted_option": 0}
                for req in slice_requests]

    # iterate chunk by chunk
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = batch_requests[start:end]
        payload = {"batch": chunk}
        attempt = 0
        chunk_success = False
        while attempt <= max_retries:
            try:
                attempt += 1
                if attempt == 1:
                    sample = chunk[0] if chunk else None
                    # print(f"[CLIENT] listener call {url} chunk={start}:{end} size={len(chunk)} sample={sample}")

                r = sess.post(url, json=payload, timeout=(connect_timeout, read_timeout))
                r.raise_for_status()
                # parse JSON
                try:
                    resp = r.json()
                except Exception as e:
                    print(f"[WARN] speaker returned non-json body for chunk {start}:{end}: {e}; body head: {r.text[:400]!r}")
                    raise

                # expected: {"results":[...], "took":...} or a list
                if isinstance(resp, dict) and "results" in resp:
                    res = resp["results"]
                elif isinstance(resp, list):
                    res = resp
                else:
                    print(f"[WARN] unexpected JSON format for chunk {start}:{end}: {type(resp)}; resp keys: {list(resp.keys()) if isinstance(resp, dict) else 'N/A'}")
                    res = neutral_resp_slice(chunk)

                # if the listener returned wrong length for this chunk, fallback for safety
                if not isinstance(res, list) or len(res) != len(chunk):
                    print(f"[WARN] speaker returned {len(res)} results but expected {len(chunk)} for chunk {start}:{end}. Using fallback for this chunk.")
                    res = neutral_resp_slice(chunk)

                results_all.extend(res)
                chunk_success = True
                break  # exit retry loop for this chunk

            except requests.exceptions.RequestException as e:
                print(f"[WARN] speaker chunk {start}:{end} request failed attempt={attempt}: {e}")
                if attempt > max_retries:
                    print(f"[ERROR] chunk {start}:{end} failed after {max_retries} retries; using neutral fallback")
                    results_all.extend(neutral_resp_slice(chunk))
                    break
                # exponential backoff
                backoff = backoff_factor * (2 ** (attempt - 1))
                time.sleep(backoff)
            except Exception as e:
                print(f"[WARN] unexpected error calling speaker for chunk {start}:{end}: {e}")
                # fallback and break
                results_all.extend(neutral_resp_slice(chunk))
                break

        # optional tiny pause to avoid bursts
        if chunk_delay and (end < total):
            time.sleep(chunk_delay)

    # final safety: ensure we return exactly one response per request
    if len(results_all) != total:
        print(f"[WARN] results length mismatch: got {len(results_all)} expected {total}; filling remainder with neutral responses")
        # pad if needed
        while len(results_all) < total:
            i = len(results_all)
            req = batch_requests[i]
            results_all.append({"descriptions": [""]*len(req.get("candidate_paths", []))})
        # trim if somehow longer
        results_all = results_all[:total]

    return results_all

# def listener_prompt()
#     pass
# ---- Corrected aggregate_speaker_requests ----
def aggregate_speaker_requests(inputs: List[Dict[str, Any]]):
    """
    Aggregate speaker requests across ranks (single-process test version).
    inputs: list of dicts, each dict should contain:
      - 'ret_paths' : list[str]
      - 'speaker_problem' : str
      - 'example_idx' or 'idx' : unique index for the example
    Returns: list of descriptions aligned with inputs order
    """
    global _speaker_cache

    # Build local requests and keys
    local_requests = []      # payloads this rank needs (for gather)
    local_indices = []       # indices into the local batch that correspond to each payload
    local_keys = []          # key for each local example (used to lookup results later)
    for i, content in enumerate(inputs):
        candidate_paths = content.get('ret_paths', [])
        names = content.get('names', [])
        # support multiple possible index field names
        sample_idx = content.get('example_idx', content.get('idx', i))
        key = (sample_idx, tuple(names))
        local_keys.append(key)
        # Only request if not present in _speaker_cache (local cache)
        if key not in _speaker_cache:
            payload = {
                "example_idx": sample_idx,
                "candidate_paths": candidate_paths,
                "names":names,
                "question": content.get('speaker_problem', "Describe the cartoon character in the image.")
            }
            local_requests.append(payload)
            local_indices.append(i)

    # Gather from other ranks (single-process stub returns [local_requests])
    gathered_requests = dist_all_gather_object_fallback(local_requests)
    world_rank, world_size = get_world_info()

    # Rank 0: dedupe and call speaker in safe chunks
    new_entries = {}
    if world_rank == 0:
        seen = set()
        unique_payloads = []
        unique_keys = []

        # flatten gathered requests and dedupe
        for rank_list in gathered_requests:
            for payload in rank_list or []:
                # Use a consistent key: (example_idx, tuple(candidate_paths))
                key = (payload["example_idx"], tuple(payload["names"]))
                if key not in seen:
                    seen.add(key)
                    unique_payloads.append(payload)
                    unique_keys.append(key)
        if unique_payloads:
            # chunk speaker calls to avoid huge requests (tune as needed)
            chunk_size = int(os.environ.get("SPEAKER_CHUNK_SIZE", "4"))
            for i in range(0, len(unique_payloads), chunk_size):
                chunk = unique_payloads[i : i + chunk_size]
                # Call the (mocked) speaker
                chunk_results = _call_speaker_batch(chunk)
                # map chunk results to keys and update local rank0 cache
                for payload_item, res in zip(chunk, chunk_results):
                    k = (payload_item["example_idx"], tuple(payload_item["names"]))
                    # store result in global cache (rank0's view)
                    _speaker_cache[k] = res
                    new_entries[k] = res
        else:
            new_entries = {}

    else:
        new_entries = None

    # Broadcast new entries from rank0 to others (single-process stub)
    new_entries = dist_broadcast_object_fallback(new_entries, src=0)

    # Merge received new entries into local speaker cache
    if new_entries:
        _speaker_cache.update(new_entries)

    # Collect descriptions in original input order
    outputs = []
    for key in local_keys:
        desc = _speaker_cache.get(key)
        if desc is None:
            # Very defensive fallback (shouldn’t happen)
            desc = ""
        outputs.append(desc)
    return outputs



def format_extra_info(extra_info: dict[str, str]) -> str:
    """Format name→description pairs as lettered options (A., B., ...)."""
    letters = list(string.ascii_uppercase)
    lines = []
    for i, (name, info) in enumerate(extra_info.items()):
        label = letters[i] if i < len(letters) else f"Option {i+1}"
        lines.append(f"{label}. Name: {name}, Info: {info}")
    return "\n".join(lines)

def modify_prompt(inputs, descriptions):
    """
    inputs[i]:  {'names': [...], 'category': str, 'image': <PIL or path>, ...}
    descriptions[i]: {'descriptions': [...]}
    Returns a new list with 'prompt' set to chat-format messages.
    """
    mod_inputs = []

    for inp, desc_item in zip(inputs, descriptions):
        names = inp.get("names", [])
        descs = desc_item.get("descriptions", [])
        
        # Align lengths safely
        n = min(len(names), len(descs))
        names, descs = names[:n], descs[:n]

        info_dict = dict(zip(names, descs))
        info_block = format_extra_info(info_dict)

        category = inp.get("category", "object")
        letters = [string.ascii_uppercase[i] for i in range(n)]
        test_question = (
            f"Which description matches the {category} in the image? "
            f"Choose the correct option from A, B, C, D or E."
        )
        # answer_format = {chr(65 + i): f"[Matching attributes for option {chr(65 + i)}]" for i in range(len(names))}
        # answer_format = {}
        # answer_format.update({
        #     "Reasoning": "<Brief justification>",
        #     "Answer": f"<one of {letters}>",
        # })
        prompt = (
            "You are a helpful AI agent specializing in image analysis and object recognition.\n"
            "You are provided with a query image along with detailed description(s) of one or several objects.\n\n"
            "Below are the description(s):\n"
            f"{info_block}\n\n"
            "Your Task:\n"
            f"- Compare the query image with each description and answer the question:\n{test_question}\n"
            "- Ignore superficial details (clothing, accessories, pose, background). Focus on non-variant/permanent features "
            "(e.g., color, shape, pattern, text for objects/buildings; facial features for people).\n"
            # "- List shared attributes between the image and each description very concisely (≤5 words).\n"
            # "- Provide a brief reasoning for your final answer."
            # "- Think briefly before generating your final answer.\n"
            f"- Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\nThe output answer format should be as follows:\n<think> ... </thnk> <answer> ... </answer>\nPlease strictly follow the format."
            # "- Respond strictly in the following JSON format:\n"
            # f"{json.dumps(answer_format, indent=2)}\n"
            # "Any deviation from this format will be considered incorrect. Do not output any additional text."
        )

        # Attach chat-style prompt. Fill the image value upstream if needed.
        inp = dict(inp)  # shallow copy so we don't mutate the original
        inp["prompt"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # provide image object/path if available
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        mod_inputs.append(inp)

    return mod_inputs

# ---- Small test runner ----
if __name__ == "__main__":
    # Example inputs: three examples, with some overlapping candidate paths
    inputs = [
        {'ret_paths': ['/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/toodles-galore/6.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/elephant/1.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/fire/4.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/marie-cat/2.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/dug/2.png'], 'speaker_problem': """Describe the cartoon character in the image so that it can be distinguished from other cartoon character objects. Do NOT mention background, location or state of the object. If the image contains a person, avoid mentioning the clothing or accesories. Write exactly one fluent sentence that begins with "The cartoon character" and highlights 3_4 visible distinguishing attributes. Keep the description concise and natural, without using lists or brackets. Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags.""", 'solution': 0, 'category': 'cartoon character', 'example_idx': '9'},
        {'ret_paths': ['/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/toodles-galore/6.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/elephant/1.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/fire/4.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/marie-cat/2.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/dug/2.png'], 'speaker_problem': """Describe the cartoon character in the image so that it can be distinguished from other cartoon character objects. Do NOT mention background, location or state of the object. If the image contains a person, avoid mentioning the clothing or accesories. Write exactly one fluent sentence that begins with "The cartoon character" and highlights 3_4 visible distinguishing attributes. Keep the description concise and natural, without using lists or brackets. Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags.""", 'solution': 0, 'category': 'cartoon character', 'example_idx': '8'},
        {'ret_paths': ['/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/toodles-galore/6.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/elephant/1.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/fire/4.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/marie-cat/2.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/dug/2.png'], 'speaker_problem': """Describe the cartoon character in the image so that it can be distinguished from other cartoon character objects. Do NOT mention background, location or state of the object. If the image contains a person, avoid mentioning the clothing or accesories. Write exactly one fluent sentence that begins with "The cartoon character" and highlights 3_4 visible distinguishing attributes. Keep the description concise and natural, without using lists or brackets. Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags.""", 'solution': 0, 'category': 'cartoon character', 'example_idx': '7'},
        {'ret_paths': ['/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/toodles-galore/6.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/elephant/1.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/fire/4.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/marie-cat/2.png', '/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/YoLLaVA/train/all/dug/2.png'], 'speaker_problem': """Describe the cartoon character in the image so that it can be distinguished from other cartoon character objects. Do NOT mention background, location or state of the object. If the image contains a person, avoid mentioning the clothing or accesories. Write exactly one fluent sentence that begins with "The cartoon character" and highlights 3_4 visible distinguishing attributes. Keep the description concise and natural, without using lists or brackets. Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags.""", 'solution': 0, 'category': 'cartoon character', 'example_idx': '6'}
    ]

    print("Initial _speaker_cache:", _speaker_cache)
    descs = aggregate_speaker_requests(inputs)
    print("Returned descriptions (aligned with inputs):")
    for i, d in enumerate(descs):
        print(f"  input[{i}] -> {d}")
    print("Final _speaker_cache keys:", list(_speaker_cache.keys()))
    