"""A small C preprocessor macro expander, enough to find kernels that macros define.

Real CUDA code often stamps out kernels with macros::

    #define DEFINE_SCALE(T, NAME) \\
        __global__ void NAME(T* x, T s, int n) { ... }
    DEFINE_SCALE(float, scale_f32)
    DEFINE_SCALE(double, scale_f64)

or marks them with an object-like macro (``#define KERNEL __global__``). A lexical
scan for ``__global__`` sees neither ``scale_f32`` nor ``scale_f64``. This module
collects ``#define`` directives and expands macro invocations the way cpp does for
the common cases: object-like and function-like macros, ``#`` stringizing, ``##``
pasting, ``__VA_ARGS__``, nested invocations and arguments that are macro names
(``FOR_TYPES(DEFINE_SCALE)``). It does not evaluate ``#if``; when a name is defined
twice the last definition wins.

Pure Python, no torch import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_DEPTH = 32
MAX_EXPANSION = 200_000  # characters; stops runaway expansions

_DEFINE = re.compile(r"#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^)]*\))?(.*)", re.S)
_TOKEN = re.compile(r'[A-Za-z_]\w*|##|#|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|\s+|\d[\w.]*|.', re.S)


@dataclass
class Macro:
    """One ``#define``."""

    name: str
    params: list[str] | None  # None for an object-like macro
    body: str  # continuations removed, newlines kept
    variadic: str  # "" (fixed arity), "va" (... / __VA_ARGS__) or "named" (GNU args...)
    line: int  # 1-based line of the #define


def strip_comments(text: str) -> str:
    """Blank out comments only (string literals kept); same length and lines as ``text``."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c, nxt = text[i], text[i + 1:i + 2]
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return "".join(out)


def directive_spans(clean: str) -> list[tuple[int, int]]:
    """(start, end) of every preprocessor directive in comment-free text, continuations included."""
    spans = []
    for m in re.finditer(r"^[ \t]*#", clean, flags=re.M):
        start = m.end() - 1
        end = start
        while True:
            j = clean.find("\n", end)
            if j < 0:
                end = len(clean)
                break
            if clean[:j].rstrip(" \t\r").endswith("\\"):
                end = j + 1
                continue
            end = j
            break
        spans.append((start, end))
    return spans


def find_macros(clean: str) -> dict[str, Macro]:
    """Every ``#define`` in comment-free text, by name (the last definition wins)."""
    macros: dict[str, Macro] = {}
    for start, end in directive_spans(clean):
        m = _DEFINE.match(re.sub(r"\\[ \t]*\r?\n", "\n", clean[start:end]))
        if not m:
            continue
        name, plist, body = m.group(1), m.group(2), m.group(3)
        params, variadic = None, ""
        if plist is not None:
            params = [p.strip() for p in plist[1:-1].split(",") if p.strip()]
            if params and params[-1].endswith("..."):
                last = params.pop()[:-3].strip()
                variadic = "named" if last else "va"
                if last:
                    params.append(last)
        macros[name] = Macro(name, params, body.strip(), variadic, clean.count("\n", 0, start) + 1)
    return macros


def split_args(text: str) -> list[str]:
    """Split macro arguments on commas outside parentheses (cpp ignores <> and {})."""
    args, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:i])
            start = i + 1
    args.append(text[start:])
    return [a.strip() for a in args]


def match_paren(text: str, i: int) -> int:
    """Index just past the ``)`` matching ``text[i] == '('``, or -1 if unbalanced."""
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


class Expander:
    """Expands macro invocations in text using a fixed set of macros."""

    def __init__(self, macros: dict[str, Macro]):
        self.macros = macros

    def expand(self, text: str, disabled: frozenset[str] = frozenset(), depth: int = 0) -> str:
        """``text`` with every macro invocation expanded (bounded by MAX_DEPTH / MAX_EXPANSION)."""
        if depth > MAX_DEPTH:
            return text
        out: list[str] = []
        size = 0
        i, n = 0, len(text)
        while i < n:
            m = _TOKEN.match(text, i)
            tok = m.group(0)
            mac = self.macros.get(tok) if tok[0].isalpha() or tok[0] == "_" else None
            if mac is None or tok in disabled:
                out.append(tok)
                size += len(tok)
                i = m.end()
                continue
            if mac.params is None:
                rep = self.expand(mac.body, disabled | {tok}, depth + 1)
                i = m.end()
            else:
                j = m.end()
                while j < n and text[j].isspace():
                    j += 1
                if j >= n or text[j] != "(":
                    out.append(tok)  # a function-like macro name without arguments is just a name
                    size += len(tok)
                    i = m.end()
                    continue
                close = match_paren(text, j)
                if close < 0:
                    out.append(text[i:])
                    break
                args = split_args(text[j + 1:close - 1])
                rep = self.expand(self.substitute(mac, args, disabled, depth), disabled | {tok}, depth + 1)
                # cpp rescans the replacement with the rest of the text: a function-like macro
                # name it ends with takes its arguments from what follows (APPLY(DEFINE)(float))
                tail = re.search(r"([A-Za-z_]\w*)\s*$", rep)
                tail_mac = self.macros.get(tail.group(1)) if tail and tail.group(1) not in disabled else None
                if tail_mac is not None and tail_mac.params is not None and text[close:].lstrip().startswith("("):
                    out.append(self.expand(rep + text[close:], disabled, depth + 1))
                    break
                i = close
            out.append(rep)
            size += len(rep)
            if size > MAX_EXPANSION:
                out.append(text[i:])
                break
        return "".join(out)

    def substitute(self, mac: Macro, args: list[str], disabled: frozenset[str], depth: int) -> str:
        """The body of ``mac`` with its parameters replaced by ``args`` (# and ## applied)."""
        params = list(mac.params or [])
        values = dict(zip(params, args + [""] * (len(params) - len(args))))
        if mac.variadic == "named":
            values[params[-1]] = ", ".join(args[len(params) - 1:])
        elif mac.variadic:
            values["__VA_ARGS__"] = ", ".join(args[len(params):])
        toks = _TOKEN.findall(mac.body)
        # 1. stringize: # param
        merged: list[str] = []
        k = 0
        while k < len(toks):
            t = toks[k]
            if t == "#":
                nxt = k + 1
                while nxt < len(toks) and toks[nxt].isspace():
                    nxt += 1
                if nxt < len(toks) and toks[nxt] in values:
                    raw = values[toks[nxt]].replace("\\", "\\\\").replace('"', '\\"')
                    merged.append('"' + " ".join(raw.split()) + '"')
                    k = nxt + 1
                    continue
            merged.append(t)
            k += 1
        # 2. parameters: raw next to ##, fully expanded elsewhere
        def neighbour_is_paste(idx: int, step: int) -> bool:
            j = idx + step
            while 0 <= j < len(merged) and merged[j].isspace():
                j += step
            return 0 <= j < len(merged) and merged[j] == "##"

        pieces: list[str] = []
        for idx, t in enumerate(merged):
            if t in values:
                if neighbour_is_paste(idx, -1) or neighbour_is_paste(idx, 1):
                    pieces.append(values[t])
                else:
                    pieces.append(self.expand(values[t], disabled, depth + 1))
            else:
                pieces.append(t)
        # 3. paste: drop ## and the whitespace around it
        body = "".join(pieces)
        return re.sub(r"\s*##\s*", "", body)
