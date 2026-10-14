import torch


def _f32(x):
    # the kernel receives these scalars as C `float`: round them like the kernel sees them
    return float(torch.tensor(x, dtype=torch.float32).item())


def reference(params_memory, grads_memory, m_memory, v_memory, learning_rate, beta1, beta2,
              beta1_correction, beta2_correction, eps, weight_decay):
    lr, b1, b2 = _f32(learning_rate), _f32(beta1), _f32(beta2)
    c1, c2, eps, wd = _f32(beta1_correction), _f32(beta2_correction), _f32(eps), _f32(weight_decay)
    g = grads_memory
    m = b1 * m_memory + (1.0 - b1) * g
    v = b2 * v_memory + (1.0 - b2) * g * g
    m_hat = m / c1
    v_hat = v / c2
    p = params_memory - lr * (m_hat / (torch.sqrt(v_hat) + eps) + wd * params_memory)
    return {"params_memory": p, "m_memory": m, "v_memory": v}
