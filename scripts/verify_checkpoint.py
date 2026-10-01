"""Verify the released DPNCA-LM checkpoint against the code in this repository.

Checks, in order:
  1. The DPNCA_160M config instantiates to exactly 158,881,044 parameters.
  2. The released fp16 safetensors file loads with strict=True (471 tensors;
     the tied output head is re-tied to the embedding at load time).
  3. Causality: perturbing the token at position t changes no logit at any
     position < t (max |delta| must be exactly 0.0).
  4. Receptive field: the structural ceiling is N * T * (k_max - 1) + 1 = 433.
  5. (optional) Cross-entropy on a UTF-8 text file, tokenized with GPT-2 BPE.
  6. (optional) A short greedy continuation from a prompt.

Usage:
    python scripts/verify_checkpoint.py --weights dpnca_160m_best_fp16.safetensors \
        [--text some_file.txt] [--prompt "The history of"] [--device cpu]
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

EXPECTED_PARAMS = 158_881_044


def load_weights(model: torch.nn.Module, path: str) -> int:
    from safetensors.torch import load_file

    sd = load_file(path)
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    if "output_head.weight" not in sd:
        sd["output_head.weight"] = sd["embedding.embedding.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    return len(sd)


@torch.no_grad()
def causality_check(model: DPNCA, device: str, length: int = 96, t: int = 60) -> float:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, length), generator=g).to(device)
    base = model(ids).float()
    ids2 = ids.clone()
    ids2[0, t:] = (ids2[0, t:] + 1) % model.config.vocab_size
    pert = model(ids2).float()
    return float((base[0, :t] - pert[0, :t]).abs().max())


@torch.no_grad()
def text_cross_entropy(model: DPNCA, device: str, text: str, seq_len: int = 1024) -> tuple[float, int]:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    toks = torch.tensor(enc.encode(text), dtype=torch.long)
    n = (len(toks) - 1) // seq_len * seq_len
    if n == 0:
        seq_len = len(toks) - 1
        n = seq_len
    toks = toks[: n + 1]
    total, count = 0.0, 0
    for i in range(0, n, seq_len):
        x = toks[i : i + seq_len].unsqueeze(0).to(device)
        y = toks[i + 1 : i + 1 + seq_len].unsqueeze(0).to(device)
        logits = model(x).float()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
        total += float(ce)
        count += y.numel()
    return total / count, count


@torch.no_grad()
def greedy_continue(model: DPNCA, device: str, prompt: str, n_new: int = 40) -> str:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(n_new):
        logits = model(ids)[0, -1].float()
        ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
    return enc.decode(ids[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--text", default=None, help="UTF-8 text file for a cross-entropy sanity check")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = DPNCA_160M
    model = DPNCA(cfg).to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[1] parameters: {n_params:,} (expected {EXPECTED_PARAMS:,}) -> {'OK' if n_params == EXPECTED_PARAMS else 'MISMATCH'}")

    n_tensors = load_weights(model, args.weights)
    print(f"[2] strict load: OK ({n_tensors} tensors incl. re-tied output head)")

    delta = causality_check(model, args.device)
    print(f"[3] causality: max |delta logit| at positions < t = {delta:.3e} -> {'OK' if delta == 0.0 else 'LEAK'}")

    rf = cfg.n_layers * cfg.n_steps * (max(cfg.perception_kernels) - 1) + 1
    print(f"[4] receptive field ceiling: {cfg.n_layers}*{cfg.n_steps}*({max(cfg.perception_kernels)}-1)+1 = {rf} tokens (context {cfg.max_seq_len})")

    if args.text:
        text = Path(args.text).read_text(encoding="utf-8")
        ce, n = text_cross_entropy(model, args.device, text)
        print(f"[5] cross-entropy on {args.text}: {ce:.4f} nats/token over {n:,} tokens (ppl {math.exp(ce):.2f})")

    if args.prompt:
        print(f"[6] greedy continuation:\n    {greedy_continue(model, args.device, args.prompt)!r}")


if __name__ == "__main__":
    main()
