"""
Step 1 of the throughput-optimization session (AUDIT.md sec T onward).

Measures peak GPU memory (torch.cuda.max_memory_allocated()) for the two
workloads that actually run on this box:
  - a real training step (forward + backward + optimizer.step(), matching
    train_qlora.py's LoRA/4-bit/paged_adamw_8bit setup) at batch sizes
    1, 2, 4, 8, with gradient_checkpointing False then True
  - a real generation call (matching LocalHFClient's 4-bit + LoRA + fp16
    setup, max_new_tokens=512) at eval batch sizes 1, 8, 16, 32, 64

Every number is MEASURED (torch.cuda.reset_peak_memory_stats() before each
config, torch.cuda.max_memory_allocated() after), never estimated. A CUDA OOM
on a given config is caught, logged as OOM, and the sweep continues -- the
whole point of the script is to find the largest OOM-free config, not to
require every config to succeed.

"Fits with >=15% headroom" = peak_allocated <= 0.85 * total GPU memory.

Usage:
    python scripts/bench_memory.py --adapter adapters/qwen-swarm-v3a
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

TRAIN_BATCH_SIZES = [1, 2, 4, 8]
GEN_BATCH_SIZES = [1, 8, 16, 32, 64]
HEADROOM_FRACTION = 0.15


def gb(nbytes: float) -> float:
    return nbytes / (1024 ** 3)


def reset_and_run(fn):
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return peak, time.time() - t0, None
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return None, time.time() - t0, "OOM"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None, time.time() - t0, "OOM"
        raise


def _one_sft_step(args, batch_size, grad_ckpt):
    """Loads a fresh model + runs exactly ONE SFTTrainer step through the REAL
    training path (SFTTrainer's own dynamic-padding collator, not a hand-rolled
    forward/backward) -- an earlier version of this script reimplemented the
    step manually with padding="max_length" and got a reproducible false OOM
    at batch=1 (22GB+ for a single 1024-token sequence) that never occurred in
    this project's actual training runs (v2/v3a/v3a-nomask/v3b/v3c all trained
    fine at batch=1 on this exact GPU minutes before this script was written).
    Going through SFTTrainer itself removes any chance of that mismatch:
    real rows are 668-813 tokens (mean 734), and SFTTrainer pads dynamically to
    the batch's actual max length, not a fixed ceiling."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)

    ds_train = load_dataset("json", data_files=args.train, split="train").select(range(batch_size * 2))

    sft_cfg = SFTConfig(
        assistant_only_loss=True,
        output_dir="/tmp/bench_memory_scratch",
        max_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds_train, processing_class=tok)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        trainer.train()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        err = None
    except torch.cuda.OutOfMemoryError:
        peak, err = None, "OOM"
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        peak, err = None, "OOM"
    dt = time.time() - t0

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return peak, dt, err


def bench_training(args, total_mem):
    """Each (grad_ckpt, batch_size) config runs in its OWN subprocess (re-invoking
    this script with --single-config) -- a fresh CUDA context per config, not just
    a fresh model object in the same process. Earlier in-process reloading hit a
    real leak: once one config OOM'd mid-step, the next config's fresh model load
    itself OOM'd (21.95 GiB already "in use" before the new model finished
    loading) -- del/gc.collect()/empty_cache() could not fully reclaim memory
    from a trainer that failed mid-backward. Cross-checking one contaminated
    result (grad_ckpt=True batch=8, which crashed while loading the NEXT config)
    against a subprocess-isolated re-measurement gave a completely different,
    much lower number (13.10 GB vs an apparent OOM) -- proof the in-process
    sweep was not trustworthy. Subprocess isolation costs a few extra seconds of
    model-reload time per config but is the only way to guarantee each number is
    real."""
    print("\n=== TRAINING benchmark: real SFTTrainer path (matches train_qlora.py exactly), "
          "one subprocess per config for full CUDA-context isolation ===")
    results = []
    for grad_ckpt in (False, True):
        for bs in TRAIN_BATCH_SIZES:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--single-config",
                  f"{grad_ckpt}:{bs}", "--model", args.model, "--train", args.train]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            peak = None
            err = None
            for line in proc.stdout.splitlines():
                if line.startswith("BENCH_RESULT:"):
                    payload = json.loads(line[len("BENCH_RESULT:"):])
                    peak, err = payload["peak_bytes"], payload["error"]
            if peak is None and err is None:
                err = "SCRIPT_ERROR"  # subprocess crashed without emitting a result line
            row = {"phase": "train", "grad_checkpointing": grad_ckpt, "batch_size": bs,
                  "peak_gb": round(gb(peak), 2) if peak else None, "seconds": round(dt, 2),
                  "error": err, "fits_15pct_headroom": (peak is not None and peak <= (1 - HEADROOM_FRACTION) * total_mem)}
            results.append(row)
            peak_str = "OOM" if err else f"{row['peak_gb']:.2f} GB"
            print(f"  train grad_ckpt={grad_ckpt} batch={bs}: {peak_str} ({dt:.1f}s)")
    return results


def run_single_config(args):
    """Invoked as a subprocess by bench_training() -- runs exactly one
    (grad_ckpt, batch_size) config in a completely fresh process/CUDA context and
    prints a single BENCH_RESULT: <json> line the parent process parses."""
    grad_ckpt_str, bs_str = args.single_config.split(":")
    grad_ckpt, bs = grad_ckpt_str == "True", int(bs_str)
    peak, dt, err = _one_sft_step(args, bs, grad_ckpt)
    print(f"BENCH_RESULT:{json.dumps({'peak_bytes': peak, 'error': err})}")


def bench_generation(args, total_mem):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("\n=== loading model for GENERATION benchmark (matches LocalHFClient) ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched causal-LM generation
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                 device_map="auto", torch_dtype=torch.float16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompt = ("You are a tactical AI analyst for a counter-UAV system. Interpret the "
             "structured sensor outputs below and give a concise tactical assessment.\n\n"
             "--- TACTICAL CONTEXT ---\nObservation window: 0.0s - 60.0s\n"
             "Dominant formation: shield\nFormation history: shield -> transitioning -> encirclement\n"
             "Return ONLY valid JSON.")
    messages = [{"role": "user", "content": prompt}]

    results = []
    for bs in GEN_BATCH_SIZES:
        batch_messages = [messages] * bs
        enc = tok.apply_chat_template(batch_messages, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt", padding=True).to(model.device)

        def gen():
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=args.max_new_tokens,
                               pad_token_id=tok.eos_token_id, do_sample=False)

        peak, dt, err = reset_and_run(gen)
        row = {"phase": "generate", "batch_size": bs,
              "peak_gb": round(gb(peak), 2) if peak else None, "seconds": round(dt, 2),
              "error": err, "fits_15pct_headroom": (peak is not None and peak <= (1 - HEADROOM_FRACTION) * total_mem)}
        results.append(row)
        peak_str = "OOM" if err else f"{row['peak_gb']:.2f} GB"
        print(f"  generate batch={bs}: {peak_str} ({dt:.1f}s)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=str(REPO / "adapters" / "qwen-swarm-v3a"))
    ap.add_argument("--train", default=str(REPO / "data" / "sft_train_final.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "bench_memory.json"))
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--single-config", default=None,
                    help=argparse.SUPPRESS)  # internal: "grad_ckpt:batch_size", used by the subprocess re-invocation in bench_training()
    args = ap.parse_args()

    if args.single_config:
        run_single_config(args)
        return

    import torch
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {torch.cuda.get_device_name(0)}, total memory: {gb(total_mem):.1f} GB, "
          f"15% headroom threshold: {gb(total_mem) * 0.85:.1f} GB")

    train_results = bench_training(args, total_mem) if not args.skip_training else []
    gen_results = bench_generation(args, total_mem) if not args.skip_generation else []
    all_results = train_results + gen_results
    Path(args.out).write_text(json.dumps(all_results, indent=2))

    print("\n\n=== MEMORY BENCHMARK SUMMARY ===")
    print("| phase | grad_checkpointing | batch | peak GB | fits 15% headroom? |")
    print("|---|---|---|---|---|")
    for r in all_results:
        gc_str = r.get("grad_checkpointing", "n/a")
        peak_str = f"{r['peak_gb']:.2f}" if r["peak_gb"] is not None else "OOM"
        fits = "YES" if r["fits_15pct_headroom"] else ("no" if r["peak_gb"] is not None else "n/a")
        print(f"| {r['phase']} | {gc_str} | {r['batch_size']} | {peak_str} | {fits} |")

    print("\nLargest batch that fits with >=15% headroom, per config:")
    for grad_ckpt in (False, True):
        fitting = [r for r in train_results if r.get("grad_checkpointing") == grad_ckpt
                  and r["fits_15pct_headroom"]]
        best = max(fitting, key=lambda r: r["batch_size"]) if fitting else None
        print(f"  train grad_checkpointing={grad_ckpt}: "
              f"{best['batch_size'] if best else 'NONE fit'}")
    fitting_gen = [r for r in gen_results if r["fits_15pct_headroom"]]
    best_gen = max(fitting_gen, key=lambda r: r["batch_size"]) if fitting_gen else None
    print(f"  generation: {best_gen['batch_size'] if best_gen else 'NONE fit'}")


if __name__ == "__main__":
    main()
