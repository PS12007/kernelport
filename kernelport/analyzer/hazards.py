"""Static warnings for CUDA patterns that tests on one GPU can miss.

One check so far: a lane can leave the kernel early (``if (tid >= C) return;``) and the
lanes left in its warp then run a warp-wide operation with the full mask
(``__shfl_xor_sync(0xFFFFFFFF, ...)``, ``__syncwarp()``, ``cg::reduce(warp, ...)``, or a
helper such as ``warpReduceSum``). That is undefined behaviour in CUDA, yet it often gives the
right answer on a given GPU, so validation alone cannot be trusted to find it. It is how
llm.c's ``softmax_forward_kernel7`` and ``softmax_forward_online_kernel8`` went wrong for
C < block size.

The check is lexical, like the rest of the analyzer. A value counts as varying within a warp
if it comes from ``threadIdx.x``, ``thread_rank()`` or ``laneid``; dividing by 32 /
``warpSize`` (or ``>> 5``) makes it the same for the whole warp, so ``warpId * 32 >= C`` is
not flagged.
"""

from __future__ import annotations

import re

LANE_SOURCES = re.compile(r"\bthreadIdx\s*\.\s*x\b|\.\s*thread_rank\s*\(\s*\)|__lane_id\s*\(|%laneid|\blaneid\b")
# operations every lane named in the mask (all 32 here) must reach
_WARP_OPS = re.compile(
    r"\b(__shfl(?:_up|_down|_xor)?_sync|__reduce_\w+_sync|__ballot_sync|__all_sync|__any_sync"
    r"|__match_\w+_sync|__syncwarp|cg::reduce|cooperative_groups::reduce)\s*\("
    r"|\.\s*(shfl(?:_up|_down|_xor)?|ballot)\s*\("
    r"|\b(\w*[Ww]arp_?[Rr]educe\w*|[Rr]educe_?[Ww]arp\w*)\s*\(")
_ASSIGN = re.compile(r"(?<![\w.\]>])([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)?(?<![=!<>])=(?!=)([^;]*);")
_DIV_WARP = re.compile(r"\s*(?:/\s*(?:32|warpSize|WARP_SIZE|kWarpSize)\b|>>\s*5\b)")


def _matching(text: str, i: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _drop_warp_uniform(expr: str) -> str:
    """Blank operands that are divided by the warp size (``tid / 32`` -> the same for the whole warp)."""
    out = expr
    for m in reversed(list(_DIV_WARP.finditer(expr))):
        end = m.start()
        k = end
        while k > 0 and out[k - 1].isspace():
            k -= 1
        if k > 0 and out[k - 1] == ")":  # (a + b) / 32
            depth, j = 0, k - 1
            while j >= 0:
                depth += out[j] == ")"
                depth -= out[j] == "("
                if depth == 0:
                    break
                j -= 1
            start = max(j, 0)
        else:  # identifier, member access or call: threadIdx.x, tid, warp.thread_rank()
            mm = re.search(r"[\w.\s]*(?:\(\s*\))?$", out[:k])
            start = mm.start() if mm else k
        out = out[:start] + " " * (m.end() - start) + out[m.end():]
    return out


def _lane_varying(body: str) -> set[str]:
    """Local names whose value can differ between lanes of one warp."""
    names: set[str] = set()
    assigns = [(m.group(1), m.group(2)) for m in _ASSIGN.finditer(body)]
    changed = True
    while changed:
        changed = False
        for name, expr in assigns:
            if name in names:
                continue
            if _varies(expr, names):
                names.add(name)
                changed = True
    return names


def _varies(expr: str, names: set[str]) -> bool:
    e = _drop_warp_uniform(expr)
    if LANE_SOURCES.search(e):
        return True
    return any(re.search(rf"(?<![\w.]){re.escape(n)}\b", e) for n in names)


def _warp_op_names(device_funcs: dict[str, str]) -> set[str]:
    """__device__ helpers that run a warp-wide operation themselves or through other helpers."""
    found = {n for n, b in device_funcs.items() if any(_needs_full_warp(b, m) for m in _WARP_OPS.finditer(b))}
    changed = True
    while changed:
        changed = False
        for n, b in device_funcs.items():
            if n not in found and any(re.search(rf"\b{re.escape(f)}\s*\(", b) for f in found):
                found.add(n)
                changed = True
    return found


_FULL_MASK = re.compile(r"^\s*(?:0x[fF]{8}[uU]?[lL]*|~\s*0[uU]?|-\s*1|\(\s*unsigned\s*\)\s*-\s*1|\w*(?:FULL|ALL)\w*)\s*$")


def _needs_full_warp(text: str, m: re.Match) -> bool:
    """False for ``__shfl_sync(mask, ...)`` with a mask that is not the full warp (e.g. ``__activemask()``)."""
    name = m.group(1)
    if not name or name in ("__syncwarp", "cg::reduce", "cooperative_groups::reduce"):
        if name == "__syncwarp":  # __syncwarp() is the full warp; __syncwarp(mask) may not be
            close = _matching(text, m.end() - 1, "(", ")")
            arg = text[m.end():close - 1] if close > 0 else ""
            return not arg.strip() or bool(_FULL_MASK.match(arg))
        return True
    close = _matching(text, m.end() - 1, "(", ")")
    if close < 0:
        return True
    args = text[m.end():close - 1]
    depth, first = 0, args
    for i, ch in enumerate(args):
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == "," and depth == 0:
            first = args[:i]
            break
    return bool(_FULL_MASK.match(first))


def _first_warp_op(text: str, helpers: set[str]) -> re.Match | None:
    best = next((m for m in _WARP_OPS.finditer(text) if _needs_full_warp(text, m)), None)
    for h in helpers:
        m = re.search(rf"\b{re.escape(h)}\s*\(", text)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def _exits(stmt: str) -> bool:
    return re.search(r"\breturn\b", stmt) is not None


def divergent_exits(body: str, device_funcs: dict[str, str] | None = None) -> list[tuple[int, str, int, str]]:
    """``(return offset, condition, warp-op offset, warp-op text)`` for each early exit that can split a warp.

    ``body`` is the kernel body with comments and strings blanked; offsets index into it.
    """
    helpers = _warp_op_names(device_funcs or {})
    lane = _lane_varying(body)
    hits = []
    for m in re.finditer(r"\bif\s*\(", body):
        c_end = _matching(body, m.end() - 1, "(", ")")
        if c_end < 0:
            continue
        cond = body[m.end():c_end - 1]
        rest = body[c_end:]
        s = len(rest) - len(rest.lstrip())
        if rest[s:s + 1] == "{":
            b_end = _matching(body, c_end + s, "{", "}")
            if b_end < 0:
                continue
            stmt, after = body[c_end + s:b_end], b_end
        else:
            semi = body.find(";", c_end)
            if semi < 0:
                continue
            stmt, after = body[c_end:semi + 1], semi + 1
        if not _exits(stmt) or not _varies(cond, lane):
            continue
        op = _first_warp_op(body[after:], helpers)
        if op is None:
            continue
        name = re.sub(r"\s*\($", "", op.group(0)).strip()
        hits.append((m.start(), " ".join(cond.split()), after + op.start(), name))
    return hits


def warnings_for(body: str, device_funcs: dict[str, str], line_of) -> list[str]:
    """Readable warnings; ``line_of(offset)`` maps a body offset to a source line."""
    out = []
    for ret, cond, op, name in divergent_exits(body, device_funcs):
        out.append(
            f"line {line_of(ret)}: `if ({cond})` lets single lanes of a warp exit, then the warp-wide "
            f"`{name}` at line {line_of(op)} expects all 32 lanes. If the condition splits a warp the "
            "result is undefined, and tests can still pass on one GPU. Test sizes that split a warp "
            "(e.g. not a multiple of 32, smaller than the block) and do not copy this into ports.")
    return out
