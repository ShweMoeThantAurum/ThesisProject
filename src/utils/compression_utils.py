"""
Provides simple sparsification and 8-bit quantization for model states.
"""
import torch

def topk_sparsify_tensor(t, sparsity):
    """Keeps top-(1-s) magnitude values and zeros the rest."""
    if sparsity <= 0.0:
        return t, 1.0
    if sparsity >= 1.0:
        return torch.zeros_like(t), 0.0
    k = max(1, int(t.numel() * (1.0 - sparsity)))
    flat = t.view(-1).abs()
    if k >= flat.numel():
        return t, 1.0
    thresh = torch.topk(flat, k).values.min()
    mask = t.abs() >= thresh
    out = t * mask
    kept_frac = mask.float().mean().item()
    return out, kept_frac

def quantize_8bit_tensor(t):
    """Quantizes a tensor to 8-bit with per-tensor scale and zero-point."""
    minv = t.min()
    maxv = t.max()
    rng = max(maxv - minv, torch.tensor(1e-8, device=t.device))
    scale = rng / 255.0
    zp = (-minv / scale).clamp(0, 255).round()
    q = ((t / scale) + zp).round().clamp(0, 255).to(torch.uint8)
    return q, scale, zp

def dequantize_8bit_tensor(q, scale, zp):
    """Dequantizes an 8-bit tensor back to float32."""
    return (q.float() - zp) * scale

def sparsify_state(state, sparsity):
    """Applies top-k sparsity to all float tensors in a state dict."""
    out = {}
    kept = []
    for k, v in state.items():
        if v.dtype.is_floating_point:
            sv, frac = topk_sparsify_tensor(v, sparsity)
            out[k] = sv
            kept.append(frac)
        else:
            out[k] = v
    kept_ratio = sum(kept) / max(1, len(kept))
    return out, kept_ratio

def quantize_state_8bit(state):
    """Quantizes float tensors in state dict to 8-bit; returns payload and meta."""
    payload, meta = {}, {}
    for k, v in state.items():
        if v.dtype.is_floating_point:
            q, s, zp = quantize_8bit_tensor(v)
            payload[k] = q
            meta[k] = (s, zp)
        else:
            payload[k] = v
            meta[k] = None
    return payload, meta

def dequantize_state_8bit(payload, meta):
    """Reconstructs float32 tensors from 8-bit payload and meta."""
    out = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and meta[k] is not None:
            s, zp = meta[k]
            out[k] = dequantize_8bit_tensor(v, s, zp).to(torch.float32)
        else:
            out[k] = v
    return out
