"""Evaluate a DPNCA-LM checkpoint on the Pile validation split with a fixed, deterministic protocol.

The training loop's in-run evaluation (`dpnca/training.py::Trainer.evaluate`) draws 20
random 1024-token windows per eval and averages per-batch perplexities. That is fine for
monitoring but has two defects for reporting a number:

  * `exp(CE)` is averaged per batch, which is >= exp(mean CE) by Jensen's inequality;
  * windows are random, so the minimum over 30 evals is a biased (optimistic) statistic.

This script instead walks N evenly spaced, non-overlapping 1024-token windows that span
the entire 25M-token validation split (the last 10% of the token cache), accumulates
token-level cross-entropy, and reports exp(mean CE). With `--windows 0` it uses every
window in the split.

Usage:
    python scripts/eval_pile_val.py --weights dpnca_160m_best_fp16.safetensors \
        --cache data_cache/pile_250M_gpt2.pt [--windows 80] [--device cuda]

The cache file is the one produced by `dpnca.data.dataset.CorpusData("pile", ..., max_tokens=250_000_000)`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpnca.config import DPNCA_160M  # noqa: E402
from dpnca.model import DPNCA  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True, help="token cache .pt (int64 tensor of 250M GPT-2 tokens)")
    ap.add_argument("--windows", type=int, default=80, help="number of evenly spaced windows; 0 = all")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from safetensors.torch import load_file

    tokens = torch.load(args.cache, map_location="cpu")
    split = int(len(tokens) * 0.9)
    val = tokens[split:].to(torch.long)
    L = args.seq_len
    n_windows_total = (len(val) - 1) // L
    if args.windows and args.windows < n_windows_total:
        starts = [int(i * (n_windows_total / args.windows)) * L for i in range(args.windows)]
    else:
        starts = [i * L for i in range(n_windows_total)]

    model = DPNCA(DPNCA_160M).to(args.device).eval()
    sd = {k: v.to(torch.float32) for k, v in load_file(args.weights).items()}
    sd.setdefault("output_head.weight", sd["embedding.embedding.weight"])
    model.load_state_dict(sd, strict=True)

    total_ce, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), args.batch):
            chunk = starts[i : i + args.batch]
            x = torch.stack([val[s : s + L] for s in chunk]).to(args.device)
            y = torch.stack([val[s + 1 : s + 1 + L] for s in chunk]).to(args.device)
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", enabled=args.device.startswith("cuda")):
                logits = model(x)
            ce = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            total_ce += float(ce)
            total_tok += y.numel()
            if (i // args.batch) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(starts)} windows  running CE {total_ce / total_tok:.4f}", flush=True)
    ce = total_ce / total_tok
    print(f"windows: {len(starts)}  tokens: {total_tok:,}")
    print(f"val cross-entropy: {ce:.4f} nats/token   perplexity exp(mean CE): {math.exp(ce):.2f}")


if __name__ == "__main__":
    main()
