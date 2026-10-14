import torch


def reference(dlogits, dlosses, probs, targets, V):
    # dlogits[b,t,v] += (probs[b,t,v] - (v == targets[b,t])) * dlosses[b,t]
    onehot = torch.nn.functional.one_hot(targets.long(), num_classes=V).to(probs.dtype)
    return {"dlogits": dlogits + (probs - onehot) * dlosses.unsqueeze(-1)}
