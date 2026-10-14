import torch


def reference(probs, targets):
    # losses[b, t] = -log(probs[b, t, targets[b, t]])
    p = torch.gather(probs, 2, targets.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(p)
