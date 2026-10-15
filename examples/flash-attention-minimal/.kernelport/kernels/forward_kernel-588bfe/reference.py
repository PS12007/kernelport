"""Reference for flash.cu forward_kernel: plain softmax attention, plus the final row max and row sum.

The kernel streams over key blocks with an online softmax and leaves in its inout buffers:
O = softmax(Q K^T * scale) V, m = rowmax(Q K^T * scale), l = rowsum(exp(Q K^T * scale - m)).
(The host starts O and l at 0 and m at -inf.)
"""

import torch


def reference(Q, K, V, softmax_scale, l, m, O):
    S = (Q @ K.transpose(-1, -2)) * softmax_scale
    row_max = S.amax(dim=-1)
    P = torch.exp(S - row_max.unsqueeze(-1))
    row_sum = P.sum(dim=-1)
    return {"O": (P / row_sum.unsqueeze(-1)) @ V, "m": row_max, "l": row_sum}
