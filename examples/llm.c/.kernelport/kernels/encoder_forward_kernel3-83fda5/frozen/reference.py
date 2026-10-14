def reference(inp, wte, wpe, B, T, C):
    return wte[inp.long()] + wpe[:T][None, :, :]
