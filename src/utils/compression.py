"""
Implements model compression baselines for communication-efficient FL:
"""

import math
import torch


# -------------------------------
# Magnitude pruning
# -------------------------------
def sparsify_state(state_dict, sparsity=0.5):
    """Zeroes smallest-magnitude weights to reach given sparsity (dense payload, fewer non-zeros)."""
    all_vals = torch.cat([v.flatten().abs() for v in state_dict.values() if v.is_floating_point()])
    k = int(len(all_vals) * sparsity)
    if k <= 0:
        return state_dict, 1.0, dense_state_size_bytes(state_dict)
    threshold = torch.topk(all_vals, k, largest=False).values.max().item()
    compressed = {}
    kept_total = total = 0
    for name, v in state_dict.items():
        if not v.is_floating_point():
            compressed[name] = v
            continue
        mask = v.abs() >= threshold
        compressed[name] = v * mask
        kept_total += mask.sum().item()
        total += mask.numel()
    kept_ratio = kept_total / max(1, total)
    # NOTE: This stays a dense tensor payload; size ~ original dense model
    return compressed, kept_ratio, dense_state_size_bytes(compressed)


# -------------------------------
# Size utilities
# -------------------------------
def dense_state_size_bytes(state_dict):
    """Dense float32 payload size in bytes."""
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.numel()
    return int(total)


def _int_index_bytes(numel, ndim):
    """
    Bytes to send indices for sparse tensors.
    We use int32 indices per non-zero element * ndim (worst-case COO-like).
    """
    # For top-k we send 1D flat indices (int32), so ndim=1 effectively.
    return 4 * numel * max(1, ndim)


# -------------------------------
# Top-k sparsification
# -------------------------------
def topk_compress_state(state_dict, k_frac=0.1):
    """
    Per-tensor Top-k sparsification.
    Returns:
      - decomp_state: dense float32 tensors (for server aggregation),
      - kept_ratio: kept / total,
      - payload_bytes: estimated compressed bytes to transmit (values + indices).
    Wire format (conceptual):
      for each tensor:
        idx:int32[k], val:float32[k]
    """
    assert 0.0 < k_frac <= 1.0, "k_frac must be in (0, 1]"
    decomp = {}
    total = kept = 0
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            # Send as-is (dense) if not float
            decomp[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        flat = t.flatten()
        numel = flat.numel()
        k = max(1, int(math.ceil(k_frac * numel)))

        # Top-k by magnitude
        vals, idxs = torch.topk(flat.abs(), k, largest=True, sorted=False)
        # get signed values
        signs = torch.sign(flat[idxs])
        top_vals = vals * signs

        # For aggregation, we reconstruct a dense tensor with zeros elsewhere
        rec = torch.zeros_like(flat)
        rec[idxs] = top_vals
        decomp[name] = rec.view_as(t)

        kept += k
        total += numel

        # Payload: indices (int32) + values (float32)
        payload_bytes += _int_index_bytes(k, ndim=1) + 4 * k  # 4 bytes/float32

    kept_ratio = kept / max(1, total)
    return decomp, kept_ratio, int(payload_bytes)


# -------------------------------
# 8-bit per-tensor symmetric quantization
# -------------------------------
def quantize8_state(state_dict):
    """
    Per-tensor symmetric int8 quantization.
    q = clamp(round(x/scale), -127..127), scale = max(|x|)/127, zero_point = 0
    Returns:
      - dequant_state: float32 dequantized tensors (for server aggregation),
      - kept_ratio: always 1.0 (same density),
      - payload_bytes: estimated compressed bytes (int8 weights + 4-byte scale per tensor).
    """
    dequant = {}
    payload_bytes = 0

    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            # Send as-is (dense)
            dequant[name] = t
            payload_bytes += t.element_size() * t.numel()
            continue

        max_abs = t.abs().max()
        if max_abs == 0:
            # All zeros: payload is trivial; still send one scale
            q = torch.zeros_like(t, dtype=torch.int8)
            scale = torch.tensor(1.0, dtype=torch.float32, device=t.device)
        else:
            scale = (max_abs / 127.0).to(torch.float32)
            q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)

        # Dequantize for server-side aggregation
        dequant[name] = (q.to(torch.float32) * scale).to(t.dtype)

        # Payload size: int8 weights + 4 bytes scale per tensor
        payload_bytes += q.numel() * 1 + 4  # 1 byte/weight + 4 bytes/scale

    kept_ratio = 1.0
    return dequant, kept_ratio, int(payload_bytes)
