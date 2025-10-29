#!/usr/bin/env python3
# lora_sanity_check.py
import argparse, json, re, sys, math
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoConfig

try:
    from peft import LoraConfig, get_peft_model
    # from peft.utils.other import get_peft_model_state_dict
except Exception as e:
    print("PEFT is required. pip install peft\nError:", e)
    sys.exit(1)


def human(n):
    return f"{n:,}"

def device_str():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()} (cap {torch.cuda.get_device_capability()})"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def list_linear_modules(model):
    # Collect names of modules that look like linear projections
    linear_types = ("Linear", "LoRA", "Lora", "LoraLinear")
    hits = []
    for name, mod in model.named_modules():
        t = mod.__class__.__name__
        if any(k in t for k in linear_types):
            hits.append((name, t))
    return hits

def find_names_by_patterns(model, patterns):
    """
    Return module names that are leaf modules whose names match any of the provided regex/prefix patterns.
    We only include modules that have weight params and look 'linear-like' (have 'weight' and are 2D).
    """
    names = []
    for name, mod in model.named_modules():
        # leaf-ish: has parameters but no children with parameters
        has_weight = any(hasattr(p, "ndim") and p.ndim == 2 for p in mod.parameters(recurse=False))
        if not has_weight:
            continue
        child_with_params = False
        for _n, _m in mod.named_children():
            if any(p.requires_grad or p.numel() > 0 for p in _m.parameters(recurse=True)):
                child_with_params = True
                break
        if child_with_params:
            continue
        # pattern match on name
        for pat in patterns:
            # treat simple tokens as substring matches; allow regex with '/' prefix
            if pat.startswith("re:"):
                if re.search(pat[3:], name):
                    names.append(name)
                    break
            else:
                if pat in name.split(".") or pat in name:
                    names.append(name)
                    break
    return sorted(set(names))

def attach_lora(model, target_modules, r, alpha, dropout, bias, task_type):
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
        target_modules=sorted(set(target_modules)),
    )
    peft_model = get_peft_model(model, cfg)
    return peft_model, cfg

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def list_lora_sites(model):
    # After PEFT wrap, LoRA modules are injected; look for their state dict keys or adapter layers
    lora_layers = []
    for n, p in model.named_parameters():
        if "lora_" in n:
            # extract module path before ".lora_..."
            path = n.split(".lora_")[0]
            lora_layers.append(path)
    return sorted(set(lora_layers))

def projector_presence_report(all_linear_names):
    # Heuristics for common projector names in VLMs
    projector_keys = ["mm_projector", "vision_proj", "visual_proj", "multimodal_projector",
                      "video_proj", "image_proj", "proj", "projector"]
    found = [n for n in all_linear_names if any(k in n for k in projector_keys)]
    return sorted(set(found))

def quick_warnings(args, target_modules, applied_sites, projector_sites):
    warns = []
    eff_scale = args.alpha / max(args.r, 1)
    if eff_scale < 0.5:
        warns.append(f"[Scaling] alpha/r = {eff_scale:.2f} (weak). Try alpha≈r or 2–4×r.")
    if args.dropout >= 0.1:
        warns.append(f"[Dropout] lora_dropout={args.dropout} can be strong for tiny batches; try 0.0–0.05.")
    if len(target_modules) == 0:
        warns.append("[Targets] No target modules matched the provided patterns.")
    if len(applied_sites) < max(1, len(target_modules) // 4):
        warns.append("[Attachment] Few LoRA sites attached compared to targets; names may not match fused linears.")
    if len(projector_sites) == 0:
        warns.append("[Projector] No projector-like modules detected. Ensure vision→LLM projector is adapted or unfrozen.")
    return warns

def main():
    ap = argparse.ArgumentParser(description="LoRA sanity check for HF causal LMs / VLMs (e.g., Qwen-VL 7B).")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2-VL-7B-Instruct", help="HF model name or local path (e.g. qwen-vl/qwen-vl-7b-instruct)")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--task_type", type=str, default="CAUSAL_LM")
    ap.add_argument("--patterns", nargs="*", default=[
        "visual.merger", "vision_model.projector", "visual.proj", "merger.mlp"
        # "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj",
        # "mm_projector", "vision_proj", "visual_proj"
    ], help="Module-name patterns to target (substring or re:<regex>)")
    ap.add_argument("--print_modules", action="store_true", help="Print all linear-like module names")
    ap.add_argument("--json_out", default="lora_sanity_summary.json")
    args = ap.parse_args()

    print("=== LoRA Sanity Check ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Device: {device_str()}")
    print(f"LoRA config: r={args.r}, alpha={args.alpha}, dropout={args.dropout}, bias={args.bias}, task_type={args.task_type}")
    print(f"Name patterns: {args.patterns}")

    # Load config first (cheaper) then model
    # cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # print(f"Loaded config: model_type={cfg.model_type}, torch_dtype={getattr(cfg, 'torch_dtype', None)}")

    # Load model weights (we keep in eval mode; training not required)
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"  # Let it automatically distribute
            )  
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded.")

    total_params, trainable_params = count_params(model)
    print(f"Base params: total={human(total_params)}, trainable={human(trainable_params)} (should be ~0 if everything is frozen)")

    # Enumerate candidate linears
    linear_hits = list_linear_modules(model)
    if args.print_modules:
        print("\n-- Linear-like modules (name :: type) --")
        for n, t in linear_hits[:5000]:
            print(f"{n} :: {t}")
        print("-- end --")

    all_linear_names = [n for n,_ in linear_hits]
    projector_sites = projector_presence_report(all_linear_names)

    # Build target modules by matching patterns
    target_modules = find_names_by_patterns(model, args.patterns)

    print(f"\nMatched {len(target_modules)} target modules from patterns.")
    # Preview a few
    for n in target_modules[:30]:
        print(" target:", n)
    if len(target_modules) > 30:
        print(f" ... (+{len(target_modules)-30} more)")

    # Attach LoRA
    model_peft, lora_cfg = attach_lora(model, target_modules, args.r, args.alpha, args.dropout, args.bias, args.task_type)
    total_after, trainable_after = count_params(model_peft)
    lora_sites = list_lora_sites(model_peft)

    print("\n=== After attaching LoRA ===")
    print(f"Total params: {human(total_after)}")
    print(f"Trainable params: {human(trainable_after)} ({100.0*trainable_after/total_after:.4f}% of total)")
    print(f"LoRA attached sites: {len(lora_sites)}")
    for n in lora_sites[:30]:
        print(" lora:", n)
    if len(lora_sites) > 30:
        print(f" ... (+{len(lora_sites)-30} more)")

    if projector_sites:
        print("\nDetected projector-like modules (heuristic):")
        for n in projector_sites:
            mark = " (LoRA attached)" if any(n==ls or n in ls for ls in lora_sites) else ""
            print(" projector:", n, mark)
    else:
        print("\nNo obvious projector-like modules detected by name heuristic.")

    # Heuristic warnings
    warns = quick_warnings(args, target_modules, lora_sites, projector_sites)
    if warns:
        print("\n=== Heuristic Warnings ===")
        for w in warns:
            print("-", w)
    else:
        print("\nNo immediate red flags detected.")

    # Summarize to JSON for easy pasting here
    summary = {
        "model": args.model_name_or_path,
        "device": device_str(),
        "lora_config": {
            "r": args.r,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "bias": args.bias,
            "task_type": args.task_type,
            "eff_scale_alpha_over_r": (args.alpha / max(args.r, 1)),
        },
        "counts": {
            "total_params": total_after,
            "trainable_params": trainable_after,
            "trainable_pct": trainable_after / max(1, total_after),
            "num_target_modules_matched": len(target_modules),
            "num_lora_sites_attached": len(lora_sites),
        },
        "sample_targets": target_modules[:100],
        "sample_lora_sites": lora_sites[:100],
        "projector_like_modules": projector_sites[:50],
        "warnings": warns,
    }
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary → {args.json_out}")
    print("Paste the JSON here and I’ll advise exact target_modules and scaling tweaks.")


if __name__ == "__main__":
    main()
