import os
from typing import List, Dict, Any, Optional
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from contextlib import nullcontext
from deepspeed.runtime.zero import GatheredParameters

# --------------------
# Simple config
# --------------------
MODEL_NAME = os.environ.get("LISTENER_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 10

# --------------------
# Small Pydantic models
# --------------------
class CandidateItem(BaseModel):
    candidate_paths: List[str]
    question: str

class BatchRequest(BaseModel):
    batch: List[CandidateItem]

class ScoreResponseItem(BaseModel):
    yes_probabilities: List[float]
    predicted_index: int

class BatchResponse(BaseModel):
    results: List[ScoreResponseItem]


# --------------------
# Model loader + minimal safe generate
# --------------------
def listener_generate_safe(model, **kwargs):
    params = [p for p in model.parameters()
              if hasattr(p, "ds_status") or "deepspeed" in str(type(p)).lower()]
    ctx = GatheredParameters(params, modifier_rank=None) if params else nullcontext()
    # inference_mode is preferred for inference safety/perf
    with torch.inference_mode():
        with ctx:
            return model.generate(**kwargs)


class ListenerService:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # get token ids for 'yes' and 'no' (we take first token id, same as your previous code)
        self.yes_token_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]
        # ensure model on chosen device
        self.model.to(device)
        self.device = next(self.model.parameters()).device

    def _prepare_image(self, path_or_obj):
        """Return either PIL.Image or path (processor accepts both)."""
        if isinstance(path_or_obj, Image.Image):
            return path_or_obj
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                raise FileNotFoundError(f"Image not found: {path_or_obj}")
            return path_or_obj
        raise ValueError("candidate must be file path or PIL.Image")

    def _build_messages_and_inputs(self, images: List[Any], question: str):
        """
        Build processor inputs for a *batch of images* with the same question.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
            for img in images
        ]

        text_list = [ self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
                     for m in messages ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # ensure tensors on same device as model
        model_device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model_device)
        return inputs
    
    def score_candidates(self, candidate_paths, question):
        """
        Score a list of candidate image paths for a single question.
        Returns dict with 'yes_probabilities' (list[float]) and 'predicted_index' (int).
        """
        # Prepare messages for each candidate (image + question)
        prepared = [self._prepare_image(p) for p in candidate_paths]
        inputs = self._build_messages_and_inputs(prepared, question)

        # Generate with output_scores so we can compute first-token logits
        outputs = listener_generate_safe(
            self.model,
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # outputs.scores is a list of tensors. scores[0] has shape [batch_size, vocab_size]
        yes_probs = []
        # if outputs.scores and len(outputs.scores) >= 1:
        first_token_logits = outputs.scores[0]  # [batch, vocab_size] on model device
        # select yes/no logits for each batch element
        yes_logits = first_token_logits[:, self.yes_token_id]
        no_logits = first_token_logits[:, self.no_token_id]
        # softmax between yes/no logits
        stacked = torch.stack([yes_logits, no_logits], dim=1)  # [batch, 2]
        probs = torch.softmax(stacked, dim=1)  # [batch, 2]
        yes_probs = probs[:, 0].detach().cpu().tolist()
        # else:
        #     # fallback: decode first token from generated sequences and check if it equals yes/no id
        #     # sequences tensor has shape [batch, seq_len]
        #     seqs = outputs.sequences.detach().cpu().tolist()
        #     # we need to compute the generated part only; compute prompt lengths from inputs.input_ids
        #     input_ids = inputs["input_ids"].detach().cpu()
        #     prompt_lens = (input_ids != self.processor.tokenizer.pad_token_id).sum(dim=1).tolist()
        #     for i, out_seq in enumerate(seqs):
        #         gen_part = out_seq[prompt_lens[i]:]
        #         if not gen_part:
        #             yes_probs.append(0.0)
        #         else:
        #             first_tok = gen_part[0]
        #             yes_probs.append(1.0 if first_tok == self.yes_token_id else 0.0)

        # # predicted index = argmax of yes_probs
        # # ensure list of floats and a valid index
        yes_probs = [float(x) for x in yes_probs]
        predicted_index = int(max(range(len(yes_probs)), key=lambda i: yes_probs[i])) if yes_probs else 0

        return {"yes_probabilities": yes_probs, "predicted_index": predicted_index}

app = FastAPI()
listener: Optional[ListenerService] = None
@app.on_event("startup")
def startup():
    global listener
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    print("[listener_service] starting up, loading model...")
    listener = ListenerService(model_name=args.model_name, device=args.device)
    print("[listener_service] ready.")

@app.post("/batch_score", response_model=BatchResponse)
def batch_score(req: BatchRequest):
    results = []
    print(f"[REQUEST]: {req}")
    for item in req.batch:
        candidate_paths = item.candidate_paths
        question = item.question
        print(f"[Question]: {question}")
        out = listener.score_candidates(candidate_paths, question)
        results.append(ScoreResponseItem(yes_probabilities=out["yes_probabilities"],
                                         predicted_index=out["predicted_index"]))
    return BatchResponse(results=results)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_device": str(LISTENER.device)}

if __name__ == "__main__":
    import argparse, traceback, sys, time, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    # simple log file capturing startup errors
    err_log = os.path.join(os.getcwd(), "listener_startup.err")
    try:
        # print some env info first (useful in SLURM logs)
        print(f"Starting listener_service on {args.host}:{args.port}")
        print("PYTHON:", sys.executable, sys.version)
        print("CUDA available:", torch.cuda.is_available())
        try:
            print("GPU device count:", torch.cuda.device_count())
            if torch.cuda.is_available():
                print("GPU name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("Failed to inspect GPU:", e)

        # If your module uses uvicorn.run(...) to start, ensure we call it with the same filename.
        # If your file defines 'app' at module level and uses uvicorn.run inside main, you can call it now.
        # Example: uvicorn.run("listener_service:app", host=args.host, port=args.port, workers=1)
        # To keep this generic without changing much, attempt to import uvicorn and run if available.
        import uvicorn
        # If your script defines the FastAPI 'app' variable, import the current module and run it.
        module_name = os.path.splitext(os.path.basename(__file__))[0]
        uvicorn.run(f"{module_name}:app", host=args.host, port=args.port, log_level="info")
    except Exception as e:
        # write full traceback to file and stdout so SLURM captures it
        tb = traceback.format_exc()
        print("FATAL ERROR during startup; writing traceback to", err_log)
        print(tb)
        with open(err_log, "w") as f:
            f.write(tb)
        # ensure non-zero exit code
        sys.exit(1)