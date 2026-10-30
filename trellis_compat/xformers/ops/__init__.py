from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Variable-length batches with at most this many sequences run one SDPA call per
# sequence; more (e.g. thousands of attention windows) are padded into one batch.
_LOOP_MAX = 16


def _to_list(x):
    return [int(v) for v in (x.tolist() if torch.is_tensor(x) else x)]


class BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen = q_seqlen
        self.kv_seqlen = kv_seqlen

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        q = _to_list(q_seqlen)
        return cls(q, q if kv_seqlen is None else _to_list(kv_seqlen))


fmha = SimpleNamespace(BlockDiagonalMask=BlockDiagonalMask)


def _sdpa(q, k, v, mask=None, scale=None):
    """q, k, v in xformers layout [B, M, H, K]."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                         attn_mask=mask, scale=scale)
    return out.transpose(1, 2)


def _pad(x, lens, length):
    """[T, H, C] packed sequences -> [B, length, H, C] plus a [B, length] validity mask."""
    lens_t = torch.tensor(lens, device=x.device)
    starts = torch.cumsum(lens_t, 0) - lens_t
    pos = torch.arange(length, device=x.device)
    valid = pos[None] < lens_t[:, None]
    idx = torch.where(valid, starts[:, None] + pos[None], 0)
    return x[idx], valid


def memory_efficient_attention(query, key, value, attn_bias=None, p=0.0, scale=None, **_):
    if attn_bias is None:
        return _sdpa(query, key, value, scale=scale)
    assert query.shape[0] == 1, "BlockDiagonalMask expects packed sequences with batch size 1"
    q, k, v = query[0], key[0], value[0]
    qs, ks = attn_bias.q_seqlen, attn_bias.kv_seqlen
    if len(qs) <= _LOOP_MAX:
        outs, qi, ki = [], 0, 0
        for lq, lk in zip(qs, ks):
            outs.append(_sdpa(q[None, qi:qi + lq], k[None, ki:ki + lk], v[None, ki:ki + lk], scale=scale)[0])
            qi += lq
            ki += lk
        return torch.cat(outs)[None]
    qp, q_valid = _pad(q, qs, max(qs))
    kp, k_valid = _pad(k, ks, max(ks))
    vp, _ = _pad(v, ks, max(ks))
    out = _sdpa(qp, kp, vp, mask=k_valid[:, None, None, :], scale=scale)
    return out[q_valid][None]
