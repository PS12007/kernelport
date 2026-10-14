"""Autotuning search space for Python ports.

A port opts in by declaring module-level constants and a ``TUNE`` dict listing
values to try for them::

    BLOCK = 1024
    NUM_WARPS = 4
    TUNE = {"BLOCK": [256, 512, 1024, 2048], "NUM_WARPS": [2, 4, 8]}

Each combination becomes a variant: the same source with those assignment lines
rewritten. Every variant is verified against the frozen reference like any port
(a block size can change correctness, not just speed) before it is timed, so a
faster variant only ever replaces the port if it passes every case.
"""

from __future__ import annotations

import ast
import itertools

MAX_VARIANTS = 32
TUNE_ROUNDS = 5  # interleaved timing rounds per passing variant
_SCALARS = (bool, int, float, str)


class TuneError(ValueError):
    """The port's TUNE declaration is missing or invalid."""


def _module_assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
    return out


def read_space(code: str) -> tuple[dict[str, list], dict]:
    """(TUNE search space, current values) of a port's source. Raises TuneError."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise TuneError(f"port has a syntax error: {exc}") from exc
    assigns = _module_assignments(tree)
    if "TUNE" not in assigns:
        raise TuneError("port declares no TUNE = {...}; add module-level constants and a TUNE dict "
                        'listing values to try, e.g. BLOCK = 1024 and TUNE = {"BLOCK": [256, 512, 1024]}')
    try:
        space = ast.literal_eval(assigns["TUNE"].value)
    except ValueError as exc:
        raise TuneError("TUNE must be a literal dict of name -> list of values") from exc
    if not isinstance(space, dict) or not space:
        raise TuneError("TUNE must be a non-empty dict of name -> list of values")
    current = {}
    for name, values in space.items():
        if not isinstance(name, str) or name == "TUNE":
            raise TuneError(f"bad TUNE key {name!r}")
        if not isinstance(values, (list, tuple)) or not values:
            raise TuneError(f"TUNE[{name!r}] must be a non-empty list")
        if not all(isinstance(v, _SCALARS) for v in values):
            raise TuneError(f"TUNE[{name!r}] values must be numbers, strings or booleans")
        if name not in assigns:
            raise TuneError(f"TUNE names {name!r} but the port has no module-level `{name} = ...`")
        try:
            current[name] = ast.literal_eval(assigns[name].value)
        except ValueError as exc:
            raise TuneError(f"`{name} = ...` must be a literal value to be tunable") from exc
    n = 1
    for values in space.values():
        n *= len(values)
    if n > MAX_VARIANTS:
        raise TuneError(f"TUNE has {n} combinations; at most {MAX_VARIANTS} are allowed")
    return {k: list(v) for k, v in space.items()}, current


def render(code: str, params: dict) -> str:
    """The port source with each ``NAME = ...`` line for ``params`` rewritten."""
    tree = ast.parse(code)
    assigns = _module_assignments(tree)
    lines = code.splitlines(keepends=True)
    # Replace bottom-up so earlier line numbers stay valid.
    for name in sorted(params, key=lambda k: assigns[k].lineno, reverse=True):
        node = assigns[name]
        first, last = node.lineno - 1, node.end_lineno - 1
        eol = "\r\n" if lines[last].endswith("\r\n") else "\n"
        tail = lines[last][node.end_col_offset:].rstrip("\r\n")
        lines[first:last + 1] = [f"{lines[first][:node.col_offset]}{name} = {params[name]!r}{tail}{eol}"]
    return "".join(lines)


def variants(code: str) -> tuple[list[dict], dict]:
    """(every parameter combination, current values). The current combination is included."""
    space, current = read_space(code)
    names = list(space)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(space[n] for n in names))]
    if current not in combos:
        combos.insert(0, current)
    return combos, current
