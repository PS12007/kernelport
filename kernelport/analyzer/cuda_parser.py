"""Find and parse CUDA ``__global__`` kernels without a compiler.

This is a careful lexical parser, not a C++ front end. It handles comments,
string literals, templates, ``__launch_bounds__``, ``extern "C"``, multi-line
signatures, pointer aliases (``T* row = out + r * n``), and ``<<<...>>>``
launch sites. Kernels embedded as strings in Python files (CuPy RawKernel,
torch load_inline) are found too, and so are kernels defined through macros
(``DEFINE_SCALE(float, scale_f32)``, ``#define KERNEL __global__``); see
``macros.py``.

Everything here is pure Python with no torch import, so it runs anywhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from kernelport.analyzer.macros import Expander, Macro, directive_spans, find_macros, match_paren, strip_comments
from kernelport.analyzer.models import KernelInfo, LaunchConfig, Param

CUDA_SUFFIXES = {".cu", ".cuh"}
MAYBE_CUDA_SUFFIXES = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx", ".inl"}
SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
             ".kernelport", "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", "site-packages"}
MAX_FILE_BYTES = 4_000_000

DTYPE_MAP = {
    "float": "float32", "double": "float64",
    "__half": "float16", "half": "float16", "at::Half": "float16", "c10::Half": "float16",
    "__nv_bfloat16": "bfloat16", "nv_bfloat16": "bfloat16", "at::BFloat16": "bfloat16",
    "c10::BFloat16": "bfloat16",
    "int": "int32", "int32_t": "int32", "unsigned": "int32", "unsigned int": "int32", "uint32_t": "int32",
    "long long": "int64", "long long int": "int64", "int64_t": "int64", "long": "int64",
    "size_t": "int64", "unsigned long long": "int64", "uint64_t": "int64",
    "short": "int16", "int16_t": "int16", "char": "int8", "signed char": "int8", "int8_t": "int8",
    "unsigned char": "uint8", "uint8_t": "uint8", "bool": "uint8",
}

FEATURE_PATTERNS: dict[str, str] = {
    "shared_memory": r"__shared__",
    "dynamic_shared_memory": r"extern\s+__shared__",
    "syncthreads": r"__syncthreads\s*\(|__syncwarp\s*\(",
    "warp_shuffle": r"__shfl\w*\s*\(",
    "warp_vote": r"__ballot_sync|__any_sync|__all_sync|__activemask",
    "atomics": r"\batomic[A-Z]\w*\s*\(",
    "fp16": r"\b__half2?\b|\bhalf2\b|__float2half|__half2float|__hadd|__hmul",
    "bf16": r"__nv_bfloat16|__bfloat162float|__float2bfloat16",
    "vector_loads": r"\b(?:float|int|uint|half|double)[234]\b|__ldg\s*\(|reinterpret_cast\s*<\s*(?:const\s+)?\w+[234]\s*\*",
    "grid_stride_loop": r"blockDim\.\w\s*\*\s*gridDim\.\w|gridDim\.\w\s*\*\s*blockDim\.\w",
    "multi_dim_indexing": r"(?:blockIdx|threadIdx)\.[yz]\b",
    "cooperative_groups": r"cooperative_groups|\bcg::",
    "tensor_cores": r"\bwmma::|mma\.sync|__hmma",
    "inline_ptx": r"\basm\s*(?:volatile\s*)?\(",
    "fast_math_intrinsics": r"__expf|__logf|__sinf|__cosf|__fdividef|__powf|__frsqrt_rn",
    "printf": r"\bprintf\s*\(",
    "textures": r"\btex[123]D\w*\s*\(|cudaTextureObject_t",
    "dynamic_parallelism": r"<<<",
}

_ASSIGN_OPS = ("<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=")
_QUALIFIERS = {"const", "volatile", "__restrict__", "__restrict", "restrict", "struct", "__grid_constant__"}


# ----------------------------------------------------------------- lexing

def strip_comments_and_strings(text: str) -> str:
    """Blank out comments and the contents of string/char literals.

    The result has the same length and line structure as ``text``, so offsets
    and line numbers stay valid.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            # raw strings R"(...)" are rare in kernels; treat like normal strings
            j = i + 1
            while j < n and text[j] != c:
                if text[j] == "\\":
                    j += 1
                elif text[j] == "\n":
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _match_close(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the bracket matching ``text[start]`` (which must be open_ch)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError(f"unbalanced {open_ch}{close_ch} starting at offset {start}")


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside (), [], {} and template <> nesting."""
    parts: list[str] = []
    depth = angle = 0
    start = 0
    for i, ch in enumerate(text):
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "<" and nxt not in "<=" and prev != "<":
            angle += 1
        elif ch == ">" and angle > 0 and prev != "-" and nxt != "=":
            angle -= 1
        elif ch == sep and depth == 0 and angle == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


# ---------------------------------------------------------------- params

def _base_type(type_text: str) -> str:
    toks = [t for t in re.split(r"[\s*&]+", type_text) if t and t not in _QUALIFIERS]
    return " ".join(toks)


def parse_param(text: str, template_types: list[str]) -> Param | None:
    """Parse one parameter declaration such as ``const float* __restrict__ x``."""
    text = " ".join(text.split())
    if not text or text == "void":
        return None
    text = re.sub(r"\s*=\s*[^,]+$", "", text)  # drop default values
    m = re.search(r"([A-Za-z_]\w*)\s*((?:\[[^\]]*\])*)$", text)
    if not m:
        return None
    name, arrays = m.group(1), m.group(2)
    type_text = text[: m.start()].strip()
    if not type_text:  # unnamed parameter: whole text is the type
        type_text, name = text, ""
    is_pointer = "*" in type_text or bool(arrays)
    if is_pointer:
        first_star = type_text.find("*") if "*" in type_text else len(type_text)
        is_const = re.search(r"\bconst\b", type_text[:first_star]) is not None
    else:
        is_const = re.search(r"\bconst\b", type_text) is not None
    base = _base_type(type_text)
    dtype = DTYPE_MAP.get(base)
    if dtype is None and base in template_types:
        dtype = "T" if base == template_types[0] else base
    return Param(name=name, type=type_text, base_type=base, is_pointer=is_pointer,
                 is_const=is_const, dtype=dtype)


def _template_type_names(template_params: list[str]) -> list[str]:
    names = []
    for tp in template_params:
        m = re.match(r"(?:typename|class)\s+([A-Za-z_]\w*)", tp)
        if m:
            names.append(m.group(1))
    return names


# ------------------------------------------------------ pointer usage

# "T* row = ...", "const float *p = ..." : a type name, then *, then the new pointer name.
_ALIAS_DECL = re.compile(
    r"\b[A-Za-z_][\w:]*\s*(?:<[^;=]*?>)?\s*\*\s*(?:__restrict__\s+|const\s+)*([A-Za-z_]\w*)\s*=(?!=)\s*([^;]*);")


def _aliases(body: str, name: str) -> set[str]:
    """Names of local pointers derived from ``name`` (e.g. ``T* row = name + off``)."""
    names = {name}
    changed = True
    while changed:
        changed = False
        for m in _ALIAS_DECL.finditer(body):
            alias, rhs = m.group(1), m.group(2)
            if alias not in names and any(re.search(rf"\b{re.escape(n)}\b", rhs) for n in names):
                names.add(alias)
                changed = True
    return names


def _usage(body: str, name: str) -> tuple[bool, bool]:
    """(read, written) for a pointer parameter, following simple aliases."""
    read = written = False
    names = _aliases(body, name)
    alias_defs = {m.start(1) for m in _ALIAS_DECL.finditer(body)}
    for n in names:
        for m in re.finditer(rf"(?<![\w.]){re.escape(n)}\b", body):
            if m.start() in alias_defs:
                continue  # the alias declaration itself
            end = m.end()
            rest = body[end:].lstrip()
            if rest.startswith("["):  # subscript: skip to after ]
                close = _match_close(body, body.index("[", end), "[", "]")
                after = body[close:].lstrip()
            else:
                after = rest
            before = body[: m.start()].rstrip()
            deref = before.endswith("*") and not re.search(r"[\w)\]]\s*\*$", before)
            if rest.startswith("[") or deref:
                if re.search(r"[(,]\s*&\s*$", before):  # &name[i] passed to atomicAdd or any call
                    written = read = True
                elif after.startswith("=") and not after.startswith("=="):
                    written = True
                elif after.startswith(_ASSIGN_OPS) or after.startswith(("++", "--")):
                    written = read = True
                else:
                    read = True
                continue
            # bare name: atomics write through it, other calls may do anything
            call = re.search(r"([A-Za-z_]\w*)\s*\(\s*&?\s*$", before)
            if call and call.group(1).startswith("atomic"):
                written = read = True
            elif re.search(r"[(,]\s*&?\s*$", before) and not re.search(r"=\s*$", before):
                written = read = True  # passed to a function: assume both
            elif re.search(r"&\s*$", before):  # &name[...] handled above; &name alone
                written = read = True
            else:
                # pointer arithmetic in an expression (e.g. rhs of an alias): not a read of data
                pass
    return read, written


# --------------------------------------------------------------- kernels

def _statement_start(clean: str, pos: int) -> int:
    """Offset where the declaration containing ``pos`` starts."""
    start = max(clean.rfind(ch, 0, pos) for ch in ";{}") + 1
    after_pp = start
    for m in re.finditer(r"^[ \t]*#[^\n]*$", clean[start:pos], flags=re.M):
        after_pp = start + m.end()  # skip past preprocessor lines
    return after_pp


def _device_functions(clean: str) -> dict[str, str]:
    """name -> body for __device__ (non-global) functions in a file."""
    funcs = {}
    for m in re.finditer(r"__device__\b[^;{}()]*?\b([A-Za-z_]\w*)\s*\(", clean):
        try:
            p_end = _match_close(clean, m.end() - 1, "(", ")")
        except ValueError:
            continue
        rest = clean[p_end:]
        mm = re.match(r"\s*(?:const\s*)?\{", rest)
        if not mm:
            continue
        b_start = p_end + mm.end() - 1
        try:
            b_end = _match_close(clean, b_start, "{", "}")
        except ValueError:
            continue
        funcs[m.group(1)] = clean[b_start:b_end]
    return funcs


def _features(body: str, device_funcs: dict[str, str], template_params: list[str]) -> list[str]:
    text = body
    seen = set()
    stack = [body]
    while stack:  # include bodies of __device__ helpers the kernel calls (transitively)
        b = stack.pop()
        for fname, fbody in device_funcs.items():
            if fname not in seen and re.search(rf"\b{re.escape(fname)}\s*\(", b):
                seen.add(fname)
                text += "\n" + fbody
                stack.append(fbody)
    feats = [f for f, pat in FEATURE_PATTERNS.items() if re.search(pat, text)]
    if template_params:
        feats.insert(0, "templates")
    return feats


def parse_source(text: str, rel_path: str, line_offset: int = 0) -> list[KernelInfo]:
    """Parse every ``__global__`` kernel definition in one source text, including macro-generated ones."""
    clean = strip_comments_and_strings(text)
    if "#" not in clean:
        kernels = _parse_kernels(text, clean, rel_path, line_offset)
    else:
        macros = find_macros(strip_comments(text))
        expander = Expander(macros)
        markers = {}  # object-like macros standing for __global__ (#define KERNEL __global__ ...)
        for name, mac in macros.items():
            if mac.params is None:
                body = expander.expand(mac.body, frozenset({name}))
                if re.search(r"\b__global__\b", body) and "{" not in body:
                    markers[name] = body
        spans = directive_spans(clean)
        kernels = _parse_kernels(text, clean, rel_path, line_offset, skip=spans, markers=markers,
                                 expander=expander if macros else None)
        kernels += _macro_kernels(text, clean, rel_path, line_offset, spans, macros, expander, markers)
        kernels.sort(key=lambda k: k.line)
    # disambiguate repeated names in one file
    counts: dict[str, int] = {}
    for k in kernels:
        counts[k.name] = counts.get(k.name, 0) + 1
    for k in kernels:
        if counts[k.name] > 1:
            k.id = f"{k.id}@{k.line}"
    return kernels


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _macro_kernels(text: str, clean: str, rel_path: str, line_offset: int, spans: list[tuple[int, int]],
                   macros: dict[str, Macro], expander: Expander, markers: dict[str, str]) -> list[KernelInfo]:
    """Kernels defined by macro invocations such as ``DEFINE_SCALE(float, scale_f32)``."""
    nocomment = strip_comments(text)
    device_funcs = _device_functions(clean)
    kernels: list[KernelInfo] = []
    consumed = 0
    names = "|".join(re.escape(n) for n in macros if n not in markers)
    if not names:
        return kernels
    for m in re.finditer(rf"\b(?:{names})\b", clean):
        if m.start() < consumed or _in_spans(m.start(), spans):
            continue
        mac = macros[m.group(0)]
        end = m.end()
        if mac.params is not None:
            j = end + len(clean[end:]) - len(clean[end:].lstrip())
            if j >= len(clean) or clean[j] != "(":
                continue
            end = match_paren(clean, j)
            if end < 0:
                continue
        expansion = expander.expand(nocomment[m.start():end])
        if not re.search(r"\b__global__\b", expansion):
            continue
        call = " ".join(text[m.start():end].split())
        exp_clean = strip_comments_and_strings(expansion)
        if "{" not in exp_clean:
            # the macro only opens a definition: KERNEL(name)(float* x) { ... }
            brace = clean.find("{", end)
            if brace < 0 or re.search(r"[;}]", clean[end:brace]):
                continue
            try:
                def_end = _match_close(clean, brace, "{", "}")
            except ValueError:
                continue
            expansion += expander.expand(nocomment[end:def_end])
            exp_clean = strip_comments_and_strings(expansion)
            end = def_end
        funcs = {**device_funcs, **_device_functions(exp_clean)}
        found = _parse_kernels(expansion, exp_clean, rel_path, 0, device_funcs=funcs)
        if not found:
            continue
        consumed = end
        line, end_line = _line_of(text, m.start()) + line_offset, _line_of(text, end) + line_offset
        for k in found:
            k.line, k.end_line, k.generated_by = line, end_line, call
            k.source = (f"// generated by {call} (macro {mac.name}, defined at line {mac.line + line_offset})\n"
                        + "\n".join(ln.rstrip() for ln in k.source.split("\n")))
            kernels.append(k)
    return kernels


def _parse_kernels(text: str, clean: str, rel_path: str, line_offset: int,
                   skip: list[tuple[int, int]] = (), markers: dict[str, str] | None = None,
                   device_funcs: dict[str, str] | None = None,
                   expander: Expander | None = None) -> list[KernelInfo]:
    """``__global__`` definitions written out in ``text`` (outside the ``skip`` spans)."""
    markers = markers or {}
    if device_funcs is None:
        device_funcs = _device_functions(clean)
    kernels: list[KernelInfo] = []
    words = "|".join(["__global__", *(re.escape(n) for n in markers)])
    for gm in re.finditer(rf"\b(?:{words})\b", clean):
        if _in_spans(gm.start(), skip):
            continue
        pos = gm.end()
        launch_bounds = ""
        if gm.group(0) in markers:
            lb = re.search(r"__launch_bounds__\s*\((.*)\)", markers[gm.group(0)], flags=re.S)
            launch_bounds = " ".join(lb.group(1).split()) if lb else ""
        # walk tokens up to the parameter list
        name = None
        while pos < len(clean):
            ws = re.match(r"\s*", clean[pos:])
            pos += ws.end()
            lb = re.match(r"__launch_bounds__\s*\(", clean[pos:])
            if lb:
                close = _match_close(clean, pos + lb.end() - 1, "(", ")")
                launch_bounds = " ".join(clean[pos + lb.end():close - 1].split())
                pos = close
                continue
            tok = re.match(r"[A-Za-z_][\w:]*", clean[pos:])
            if tok:
                name = tok.group(0)
                pos += tok.end()
                continue
            break
        if not name or pos >= len(clean) or clean[pos] != "(":
            continue
        try:
            p_end = _match_close(clean, pos, "(", ")")
        except ValueError:
            continue
        after = re.match(r"\s*", clean[p_end:])
        b_start = p_end + after.end()
        if b_start >= len(clean) or clean[b_start] != "{":
            continue  # declaration only
        b_end = _match_close(clean, b_start, "{", "}")

        stmt_start = _statement_start(clean, gm.start())
        prefix = clean[stmt_start:gm.start()]
        template_params: list[str] = []
        tm = re.search(r"template\s*<", prefix)
        if tm:
            t_open = stmt_start + tm.end() - 1
            t_close = _match_close(clean, t_open, "<", ">")
            template_params = split_top_level(clean[t_open + 1:t_close - 1])
        extern_c = re.search(r'extern\s*"C"', text[stmt_start:gm.start()]) is not None
        src_start = stmt_start + (len(prefix) - len(prefix.lstrip()))

        ttypes = _template_type_names(template_params)
        params = []
        for raw in split_top_level(clean[pos + 1:p_end - 1]):
            p = parse_param(raw, ttypes)
            if p is not None:
                params.append(p)
        body = clean[b_start:b_end]
        if expander is not None:  # see through macros used in the body (KERNEL_LOOP(i, n) ...)
            body = expander.expand(body)
        for p in params:
            if p.is_pointer and p.name:
                p.read, p.written = _usage(body, p.name)

        kernels.append(KernelInfo(
            id=f"{rel_path}::{name}",
            name=name,
            file=rel_path,
            line=_line_of(text, gm.start()) + line_offset,
            end_line=_line_of(text, b_end - 1) + line_offset,
            params=params,
            template_params=template_params,
            launch_bounds=launch_bounds,
            extern_c=extern_c,
            features=_features(clean[pos:p_end] + "\n" + body, device_funcs, template_params),
            source=text[src_start:b_end],
        ))
    return kernels


_LAUNCH_RE = re.compile(
    r"\b([A-Za-z_][\w:]*)\s*(?:<((?:[^<>;]|<[^<>;]*>)*)>)?\s*<<<"
)


def find_launches(text: str, rel_path: str, line_offset: int = 0) -> list[tuple[str, LaunchConfig]]:
    """All ``name<<<...>>>(...)`` launch sites as (kernel name, LaunchConfig)."""
    clean = strip_comments_and_strings(text)
    out = []
    for m in _LAUNCH_RE.finditer(clean):
        cfg_start = m.end()
        cfg_end = clean.find(">>>", cfg_start)
        if cfg_end < 0:
            continue
        cfg = split_top_level(clean[cfg_start:cfg_end])
        rest = re.match(r"\s*\(", clean[cfg_end + 3:])
        args: list[str] = []
        if rest:
            a_open = cfg_end + 3 + rest.end() - 1
            try:
                a_close = _match_close(clean, a_open, "(", ")")
                args = [" ".join(a.split()) for a in split_top_level(text[a_open + 1:a_close - 1])]
            except ValueError:
                pass
        cfg_src = split_top_level(text[cfg_start:cfg_end])
        if len(cfg_src) != len(cfg):
            cfg_src = cfg
        cfg_src = [" ".join(c.split()) for c in cfg_src]
        name = m.group(1).split("::")[-1]
        out.append((name, LaunchConfig(
            file=rel_path, line=_line_of(text, m.start()) + line_offset,
            grid=cfg_src[0] if len(cfg_src) > 0 else "",
            block=cfg_src[1] if len(cfg_src) > 1 else "",
            shared_mem=cfg_src[2] if len(cfg_src) > 2 else "0",
            stream=cfg_src[3] if len(cfg_src) > 3 else "",
            template_args=" ".join((m.group(2) or "").split()),
            args=args,
        )))
    return out


def _python_cuda_strings(text: str) -> list[tuple[str, int]]:
    """(source, line offset) for string literals in Python code that contain __global__."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__global__" in node.value:
            # string starts on node.lineno; content usually begins on that line or the next
            found.append((node.value, node.lineno - 1))
    return found


# ------------------------------------------------------------------ scan

def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        if p.is_file() and (p.suffix in CUDA_SUFFIXES or p.suffix in MAYBE_CUDA_SUFFIXES or p.suffix == ".py"):
            yield p


def scan(path: Path) -> list[KernelInfo]:
    """Find and parse every kernel under ``path`` (a file or directory)."""
    root = Path(path).resolve()
    base = root.parent if root.is_file() else root
    kernels: list[KernelInfo] = []
    launches: list[tuple[str, LaunchConfig]] = []
    for f in _iter_files(root):
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "__global__" not in text and "<<<" not in text:
            continue
        rel = f.relative_to(base).as_posix()
        if f.suffix == ".py":
            for src, off in _python_cuda_strings(text):
                kernels.extend(parse_source(src, rel, off))
            continue
        kernels.extend(parse_source(text, rel))
        launches.extend(find_launches(text, rel))
    by_name: dict[str, list[KernelInfo]] = {}
    for k in kernels:
        by_name.setdefault(k.name, []).append(k)
    for name, cfg in launches:
        for k in by_name.get(name, []):
            k.launches.append(cfg)
    return kernels
