"""Reproduce the DPNCA_160M parameter count (158,881,044) from the config."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from dpnca.config import DPNCA_160M
from dpnca.model import DPNCA

model = DPNCA(DPNCA_160M)
counts = model.count_parameters()
print("=== DPNCA-160M Parameter Breakdown ===")
for k, v in counts.items():
    print(f"  {k}: {v:>15,}")

cfg = DPNCA_160M
print(f"\nd_model={cfg.d_model}, n_layers={cfg.n_layers}, "
      f"R={cfg.n_compute_rows}, C={cfg.n_channels}, "
      f"T={cfg.n_steps}, M={cfg.n_modes}, "
      f"C_H={cfg.n_hw_channels}, d_ff={cfg.d_ff}")
print(f"max_seq_len={cfg.max_seq_len}")
print(f"hw_mod_interval={cfg.hw_mod_interval}, n_steps={cfg.n_steps}")
print(f"perception_kernels={cfg.perception_kernels}")

eff = 4 * 1024 * 8
print(f"\nEffective tokens/step = 4 x 1024 x 8 = {eff:,}")
print(f"Total steps = 7630, total tokens = {7630 * eff / 1e6:.0f}M")

# VRAM estimate
import torch
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"\nModel fp32 size: {param_bytes / 1024**2:.0f} MB")
print(f"Optimizer states (AdamW): ~{param_bytes * 2 / 1024**2:.0f} MB")
print(f"Estimated VRAM (model+opt+grad): ~{param_bytes * 4 / 1024**3:.1f} GB")
