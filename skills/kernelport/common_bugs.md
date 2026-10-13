# Common porting bugs (all backends)

Read this once before your first port. When `verify_port` fails, match the
failure report against these symptoms first.

| Symptom in the report | Likely cause |
|---|---|
| `unexpected NaN` at the end of the tensor, odd sizes only | Tail not covered: grid rounded down, or loop stops at `n - n % BLOCK` |
| `unexpected NaN` everywhere | Port never writes the output (wrote to a copy, returned a new tensor, wrong param name) |
| `unexpected NaN` in a few rows | 0/0 or inf-inf: missing max-subtraction (softmax), uninitialised accumulator, masked lanes with a non-neutral `other=` |
| Fails only for rows longer than one block (e.g. cols=4097, not 1024) | Chunk loop missing or reduction only covers the first chunk |
| Fails only for tiny sizes (1, 7) | Block size larger than data without a mask, or division by a size that becomes 0 |
| Fails only for fp16 / bf16 | Accumulating in half precision; missing `.to(float32)` before math |
| Fails only for float32 with ~1e-3 error | TF32 / fast-math / reduced-precision instructions |
| Every value off by a constant factor | Wrong normalisation: divide by `n` vs `n-1`, eps inside vs outside sqrt |
| Values correct but in the wrong place | Transposed indexing: row/col swapped, wrong stride, wrong output layout |
| Off by exactly one element / shifted | Inclusive vs exclusive scan, off-by-one in offsets |
| `inout` output wrong | Overwrote instead of updating (`y = a*x` instead of `y = a*x + y`), or ignored the initial value |
| Integer outputs differ | Wrong rounding/truncation, int32 overflow in index math |
| Crash / illegal address | Unmasked out-of-bounds load, index array value used without bounds |

## Checklist before calling verify_port

- [ ] Same math as the CUDA original, including its exact formula quirks (where eps goes,
      biased variance, clamping, rounding mode of integer division).
- [ ] Every element of every output is written exactly once (or accumulated, for inouts/atomics).
- [ ] Bounds handled for every size in the spec: 1, 7, odd, power of two, large.
- [ ] Reductions accumulate in float32 for fp16/bf16 inputs.
- [ ] Index math cannot overflow 32 bits for the large case.
- [ ] No shortcuts that skip the kernel (calling torch / numpy on the data instead of the backend).

## When you are stuck

After 2 failures with the same symptom, change strategy: simplify the port (one
element per lane, no tiling), get it to PASS, then optimise and verify again.
After 6 failed attempts total, stop and report FAIL with your best explanation.
