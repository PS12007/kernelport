"""Tiny, safe integer expression evaluator for launch specs.

Shapes, grid sizes and scalar values in launch.json are written as strings
like ``"cdiv(n, 256)"`` or ``"rows * cols"``. They are evaluated against the
case's dims. Only arithmetic on names and numbers plus a few helper functions
is allowed; anything else raises ValueError.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


_FUNCS = {
    "cdiv": lambda a, b: -(-a // b),
    "min": min,
    "max": max,
    "next_pow2": _next_pow2,
    "ceil": math.ceil,
    "floor": math.floor,
    "int": int,
}


def evaluate(expr: Any, dims: dict[str, Any]) -> Any:
    """Evaluate ``expr`` (a number or expression string) with ``dims`` as variables."""
    if isinstance(expr, (int, float)):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"expression must be a number or string, got {expr!r}")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body, dims)


def _eval(node: ast.AST, dims: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in dims:
            raise ValueError(f"unknown dimension {node.id!r}")
        return dims[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, dims), _eval(node.right, dims))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand, dims)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        return _FUNCS[node.func.id](*[_eval(a, dims) for a in node.args])
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def evaluate_int(expr: Any, dims: dict[str, Any]) -> int:
    """Evaluate and require an integer result."""
    v = evaluate(expr, dims)
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"{expr!r} evaluated to non-integer {v}")
        v = int(v)
    return int(v)
