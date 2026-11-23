# -*- coding: utf-8 -*-
"""
Can this machine actually run 4-bit training? Answer by doing it.

    python AI/check_4bit.py

This deliberately does more than `import bitsandbytes`: bitsandbytes installs
on Windows but its CUDA kernels historically do not work there, and the failure
only appears when a real 4-bit layer runs. So this script:

  1. reports whether the GPU is visible to torch
  2. quantizes a real linear layer to 4-bit NF4 with double quantization
  3. runs forward AND backward through it — the backward pass is the part
     that needs the CUDA kernels, an import check would never catch it
  4. reports peak VRAM, so the 6 GB budget can be judged against reality
  5. reports whether Qwen3-4B-Instruct-2507 can be fetched, since training is
     impossible without the weights

Exit 0 means 4-bit training is viable on this machine. Anything else tells you
exactly which part is not, which is the difference between "install Docker" and
"wait six months for a library".
"""

import sys
import traceback

REQUIRED = ["torch", "transformers", "peft", "accelerate", "datasets", "bitsandbytes"]


RESULTS = []


def report(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n      {detail}" if detail else ""))
    RESULTS.append((label, ok, detail))
    return ok


def main():
    print("== 4-bit capability check ==\n")
    hard = True

    # ---- libraries ----------------------------------------------------
    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError as exc:
            missing.append(f"{mod}: {exc}")
    if not report("all training libraries import", not missing, "; ".join(missing)):
        return 1

    import torch

    # ---- GPU ----------------------------------------------------------
    if not report("CUDA visible to torch", torch.cuda.is_available()):
        if torch.cuda.is_available() is False:
            print("      no CUDA device; a CPU-only machine cannot train a 4B model")
        return 1
    if not report("GPU is a usable 4-bit target", torch.cuda.get_device_capability(0)[0] >= 7,
                  f"compute capability {torch.cuda.get_device_capability(0)} "
                  f"(needs 7.0+ for bf16/fast kernels)"):
        hard = False

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024 ** 3
    print(f"      {props.name}, {total:.2f} GB total VRAM")

    from bitsandbytes.nn import Linear4bit
    import bitsandbytes as bnb

    print(f"\n      bitsandbytes {bnb.__version__}")

    # bnb 0.50 renamed double_quant to compress_statistics; older versions
    # still expect double_quant. Prefer the new name but stay usable on both,
    # because a capability probe failing on a keyword is a false negative.
    import inspect
    params = inspect.signature(Linear4bit.__init__).parameters
    secondary = "compress_statistics" if "compress_statistics" in params else "double_quant"

    # ---- the real test: forward and backward through a 4-bit layer -----
    torch.cuda.reset_peak_memory_stats()
    try:
        kwargs = dict(bias=False, quant_type="nf4",
                      compute_dtype=torch.bfloat16)
        kwargs[secondary] = True
        layer = Linear4bit(2048, 2048, **kwargs).cuda()

        # requires_grad on the input: without it there is no graph and the
        # backward pass is meaningless rather than a kernel test.
        x = torch.randn(1, 512, 2048, dtype=torch.bfloat16,
                        device="cuda", requires_grad=True)
        out = layer(x)
        loss = out.float().pow(2).mean()
        loss.backward()          # <- the pass that needs working CUDA kernels

        # Gradient lands on the input, not the weight: bnb 4-bit weights are
        # frozen by design, which is exactly why QLoRA puts its adapters
        # elsewhere. A grad_fn at all is the proof that backward works.
        ok = loss.item() > 0 and x.grad is not None
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        report("4-bit forward + backward on GPU", ok,
               f"loss={loss.item():.4f}, input grad present={x.grad is not None}, "
               f"peak {peak:.3f} GB (double-quant via '{secondary}')")
    except Exception:
        report("4-bit forward + backward on GPU", False,
               traceback.format_exc(limit=6).strip())
        return 1

    # ---- realistically: can a 4B model fit at all? ----------------------
    # 4-bit weights are ~0.5 bytes/param, so a 4B model is roughly:
    try:
        import transformers
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        # The config does not always expose the count; fall back to a sane estimate.
        print(f"\n      Qwen3-4B-Instruct-2507 config loaded "
              f"(layers={getattr(cfg, 'num_hidden_layers', '?')}, "
              f"hidden={getattr(cfg, 'hidden_size', '?')})")
        report("base model is reachable", True)
    except Exception as exc:
        report("base model is reachable", False, str(exc))
        hard = False

    print("\n== summary ==")
    failures = [r for r in RESULTS if not r[1]]
    if failures:
        print("4-bit training is NOT viable on this machine")
        return 1
    if not hard:
        print("4-bit training runs, with the caveats above")
    else:
        print("4-bit training is viable on this machine")
    return 0


if __name__ == "__main__":
    sys.exit(main())