"""Static warnings for CUDA patterns that tests on one GPU can miss.

Two checks.

Warp: a lane can leave the kernel early (``if (tid >= C) return;``) and the
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

Block: a block-wide barrier (``__syncthreads()``, ``block.sync()``, or a helper such as
``blockReduce``) that some threads of the block never reach, because they returned before it
(``if (i >= n) return; ... __syncthreads();``) or because it sits in a branch or loop whose
condition differs between threads. CUDA requires the whole block to reach the barrier; on
recent GPUs the early-return form usually works anyway. It matters most for ports: WGSL
compilers reject ``workgroupBarrier()`` outside uniform control flow, and a Metal
``threadgroup_barrier`` that some threads skip can hang. Here any value derived from
``threadIdx`` varies, even after dividing by 32.
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


# ---- block-wide barriers -------------------------------------------------------------------

BLOCK_SOURCES = re.compile(
    r"\bthreadIdx\s*\.\s*[xyz]\b|\.\s*thread_(?:rank|index)\s*\(\s*\)|__lane_id\s*\(|%laneid|\blaneid\b")
_BARRIERS = re.compile(
    r"\b(__syncthreads(?:_count|_and|_or)?|__barrier_sync|cg::sync|cooperative_groups::sync)\s*\("
    r"|\b(this_thread_block\s*\(\s*\)\s*\.\s*sync)\s*\("
    r"|\b(\w*[Bb]lock_?[Rr]educe\w*|[Rr]educe_?[Bb]lock\w*)\s*(?:<[^;(){}]*>)?\s*\(")
_TEMPLATE_ARGS = r"\s*(?:<[^;(){}]*>)?\s*\("


def _block_varying(body: str, sources: re.Pattern = BLOCK_SOURCES) -> set[str]:
    names: set[str] = set()
    assigns = [(m.group(1), m.group(2)) for m in _ASSIGN.finditer(body)]
    changed = True
    while changed:
        changed = False
        for name, expr in assigns:
            if name not in names and _block_varies(expr, names, sources):
                names.add(name)
                changed = True
    return names


def _block_varies(expr: str, names: set[str], sources: re.Pattern = BLOCK_SOURCES) -> bool:
    if sources.search(expr):
        return True
    return any(re.search(rf"(?<![\w.]){re.escape(n)}\b", expr) for n in names)


def _barrier_names(device_funcs: dict[str, str], barriers: re.Pattern = _BARRIERS) -> set[str]:
    """__device__ helpers that reach a block barrier themselves or through other helpers."""
    found = {n for n, b in device_funcs.items() if barriers.search(b)}
    changed = True
    while changed:
        changed = False
        for n, b in device_funcs.items():
            if n not in found and any(re.search(rf"\b{re.escape(f)}{_TEMPLATE_ARGS}", b) for f in found):
                found.add(n)
                changed = True
    return found


def _first_barrier(text: str, helpers: set[str], blocks: set[str],
                   barriers: re.Pattern = _BARRIERS) -> re.Match | None:
    """First barrier in ``text``: a builtin, ``block.sync()`` on a thread_block, or a helper that has one."""
    cands = [m for m in [barriers.search(text)] if m]
    for b in blocks:
        m = re.search(rf"\b{re.escape(b)}\s*\.\s*sync\s*\(", text)
        if m:
            cands.append(m)
    for h in helpers:
        m = re.search(rf"\b{re.escape(h)}{_TEMPLATE_ARGS}", text)
        if m:
            cands.append(m)
    return min(cands, key=lambda m: m.start(), default=None)


def _op_name(m: re.Match) -> str:
    return re.sub(r"\s*(?:<[^;(){}]*>)?\s*\($", "", m.group(0)).strip()


def _stmt_end(body: str, i: int) -> int:
    """End offset of the statement starting at ``i`` (a block, if/else chain, loop or simple statement)."""
    n = len(body)
    while i < n and body[i].isspace():
        i += 1
    if i >= n:
        return n
    if body[i] == "{":
        e = _matching(body, i, "{", "}")
        return e if e > 0 else n
    m = re.match(r"(if|for|while|switch)\s*(?:constexpr\s*)?\(", body[i:])
    if m:
        c_end = _matching(body, i + m.end() - 1, "(", ")")
        if c_end < 0:
            return n
        end = _stmt_end(body, c_end)
        if m.group(1) == "if":
            em = re.match(r"\s*else\b", body[end:])
            if em:
                end = _stmt_end(body, end + em.end())
        return end
    m = re.match(r"do\b", body[i:])
    if m:
        end = _stmt_end(body, i + m.end())
        sm = body.find(";", end)
        return sm + 1 if sm >= 0 else n
    semi = body.find(";", i)
    return semi + 1 if semi >= 0 else n


def _loop_cond(head: str, kind: str) -> str:
    if kind != "for":
        return head
    parts = head.split(";")
    return parts[1] if len(parts) == 3 else head


def _varying_terms(expr: str, names: set[str]) -> list[str]:
    terms = [" ".join(m.group(0).split()) for m in BLOCK_SOURCES.finditer(expr)]
    terms += [n for n in names if re.search(rf"(?<![\w.]){re.escape(n)}\b", expr)]
    return sorted(terms)


def _same_trip_count(head: str, names: set[str]) -> bool:
    """``for (int t = base + tid; t < end + tid; t += C)``: start and bound shift by the same
    per-thread term, so every thread runs the loop the same number of times."""
    parts = head.split(";")
    if len(parts) != 3:
        return False
    init = re.match(r"\s*(?:[\w:<>]+\s+)*([A-Za-z_]\w*)\s*=(?!=)(.*)$", parts[0], re.S)
    cond = re.match(r"\s*([A-Za-z_]\w*)\s*(?:<=?|!=)(.*)$", parts[1], re.S)
    if not init or not cond or init.group(1) != cond.group(1):
        return False
    start, bound = _varying_terms(init.group(2), names), _varying_terms(cond.group(2), names)
    return bool(bound) and start == bound


def divergent_barriers(body: str, device_funcs: dict[str, str] | None = None, *,
                       sources: re.Pattern = BLOCK_SOURCES, barriers: re.Pattern = _BARRIERS,
                       strict: bool = False) -> list[tuple[str, int, str, int, str]]:
    """``(kind, offset, condition, barrier offset, barrier text)`` for each barrier some threads can miss.

    ``kind`` is ``"exit"`` (threads returned before the barrier), ``"branch"`` (barrier inside an
    if/else whose condition varies across the block) or ``"loop"`` (inside a loop whose trip count
    varies). Only the outermost divergent construct around a barrier is reported.

    ``strict`` follows WGSL's uniformity rules as browsers (Tint) apply them: no exemption for loops
    whose start and bound shift by the same thread index, and any thread-dependent return before a
    barrier counts, even inside a branch that returns anyway (both measured rejected in Edge).
    """
    helpers = _barrier_names(device_funcs or {}, barriers)
    blocks = set(re.findall(r"\bthread_block\s+(\w+)\s*=", body))
    var = _block_varying(body, sources)
    hits: list[tuple[str, int, str, int, str]] = []
    covered_until = -1
    for m in re.finditer(r"\b(if|for|while)\s*(?:constexpr\s*)?\(", body):
        if m.start() < covered_until or (m.start() > 0 and body[m.start() - 1] in "_."):
            continue
        c_end = _matching(body, m.end() - 1, "(", ")")
        if c_end < 0:
            continue
        kind = m.group(1)
        head = body[m.end():c_end - 1]
        cond = head if kind == "if" else _loop_cond(head, kind)
        if not _block_varies(cond, var, sources):
            continue
        if kind == "for" and not strict and _same_trip_count(head, var):
            continue
        end = _stmt_end(body, m.start())
        inner = body[c_end:end]
        cond_txt = " ".join(cond.split())
        bar = _first_barrier(inner, helpers, blocks, barriers)
        if bar is not None:
            hits.append(("branch" if kind == "if" else "loop", m.start(), cond_txt, c_end + bar.start(), _op_name(bar)))
            covered_until = end
            continue
        if kind == "if" and _exits(inner):
            at = _barrier_after(body, end, helpers, blocks, barriers, strict)
            if at is not None:
                hits.append(("exit", m.start(), cond_txt, at[0], at[1]))
                covered_until = len(body)  # one exit warning per kernel is enough
    return hits


def _barrier_after(body: str, pos: int, helpers: set[str], blocks: set[str],
                   barriers: re.Pattern = _BARRIERS, strict: bool = False) -> tuple[int, str] | None:
    """First barrier a thread can reach after ``pos``, walking out of the enclosing blocks.

    Stops at a block that always ends in ``return;``: threads in it never get further
    (not in ``strict`` mode, where any later barrier counts).
    """
    if strict:
        bar = _first_barrier(body[pos:], helpers, blocks, barriers)
        return (pos + bar.start(), _op_name(bar)) if bar else None
    while pos < len(body):
        depth, close = 0, len(body)
        for j in range(pos, len(body)):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                if depth == 0:
                    close = j
                    break
                depth -= 1
        bar = _first_barrier(body[pos:close], helpers, blocks, barriers)
        if bar is not None:
            return pos + bar.start(), _op_name(bar)
        if close >= len(body):
            return None
        depth, j = 0, close
        while j >= 0:  # the block's opening brace
            depth += body[j] == "}"
            depth -= body[j] == "{"
            if depth == 0:
                break
            j -= 1
        if re.search(r"\breturn\b[^;{}]*;\s*$", body[j + 1:close]):
            return None
        pos = close + 1
    return None


def warnings_for(body: str, device_funcs: dict[str, str], line_of) -> list[str]:
    """Readable warnings; ``line_of(offset)`` maps a body offset to a source line."""
    out = []
    for ret, cond, op, name in divergent_exits(body, device_funcs):
        out.append(
            f"line {line_of(ret)}: `if ({cond})` lets single lanes of a warp exit, then the warp-wide "
            f"`{name}` at line {line_of(op)} expects all 32 lanes. If the condition splits a warp the "
            "result is undefined, and tests can still pass on one GPU. Test sizes that split a warp "
            "(e.g. not a multiple of 32, smaller than the block) and do not copy this into ports.")
    port_fix = ("In ports, keep every thread running to the barrier and guard the work instead "
                "(e.g. `if (i < n) { ... }`).")
    for kind, at, cond, op, name in divergent_barriers(body, device_funcs):
        where = f"line {line_of(at)}"
        if kind == "exit":
            out.append(
                f"{where}: `if ({cond})` lets some threads of the block return before the block barrier "
                f"`{name}` at line {line_of(op)}. NVIDIA GPUs count exited threads as arrived, so the "
                "original works, but a port must not copy it: WGSL rejects `workgroupBarrier()` after a "
                f"non-uniform return, and Metal requires every thread to reach `threadgroup_barrier`. {port_fix}")
            continue
        if kind == "branch":
            what = f"the block barrier `{name}` at line {line_of(op)} sits inside `if ({cond})`, which differs between threads"
        else:
            what = (f"the block barrier `{name}` at line {line_of(op)} sits inside a loop whose condition "
                    f"`{cond}` differs between threads, so they can run it a different number of times")
        out.append(
            f"{where}: {what}. CUDA requires every live thread of the block to reach the same barrier; "
            "otherwise it can hang or let threads read shared memory too early, and tests may still pass "
            "on one GPU. Test sizes where the condition splits a block (e.g. not a multiple of the block). "
            f"WGSL rejects a barrier in such control flow. {port_fix}")
    return out
