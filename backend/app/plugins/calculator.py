"""Calculator as a tool provider (#54).

Gives the model one tool - ``calculate`` - that evaluates an arithmetic
expression with real code, so it never has to do multi-digit or multi-step math
in its head (qwen3:8b is unreliable at that, which is the whole bug). Always
enabled; no external service.

Evaluation is a locked-down **AST walk**, not ``eval``: only numeric literals,
the arithmetic operators, parentheses, a small whitelist of math functions, and a
few constants are allowed. Names, attribute access, calls to anything else, and
oversized power/factorial operands are rejected - so a garbled or hostile
expression returns an error instead of executing code or hanging.
"""

from __future__ import annotations

import ast
import calendar
import math
import operator
from datetime import date, datetime

from .. import timers  # for the configured local timezone ("today")

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_MAX_POW = 1000        # reject 10**10**10-style blowups
_MAX_FACTORIAL = 1000  # factorial grows absurdly fast


class _Unsafe(Exception):
    """Raised for anything outside the arithmetic whitelist."""


def _factorial(n):
    if not isinstance(n, int) or n < 0 or n > _MAX_FACTORIAL:
        raise _Unsafe("factorial needs a non-negative integer ≤ 1000")
    return math.factorial(n)


_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "min": min, "max": max, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp, "factorial": _factorial, "gcd": math.gcd,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _ev(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _Unsafe("only numbers allowed")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(_ev(node.right)) > _MAX_POW:
            raise _Unsafe("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        if node.keywords:
            raise _Unsafe("keyword arguments not allowed")
        return _FUNCS[node.func.id](*(_ev(a) for a in node.args))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise _Unsafe("unsupported expression")


def evaluate(expression: str):
    """Evaluate an arithmetic expression safely. Raises ``_Unsafe``/``SyntaxError``
    (and the usual math errors) on anything invalid."""
    return _ev(ast.parse(expression, mode="eval").body)


def _to_date(s: str) -> date:
    """Parse an ISO date (``YYYY-MM-DD``, optionally with a time) into a ``date``."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")
    return date.fromisoformat(s[:10])


def date_diff(start: date, end: date):
    """Exact whole-day count and a calendar years/months/days breakdown for
    ``end - start`` (total is negative when end precedes start). Calendar-correct
    across months of different lengths - the thing the model gets wrong by hand."""
    total = (end - start).days
    a, b = (start, end) if start <= end else (end, start)
    years, months, days = b.year - a.year, b.month - a.month, b.day - a.day
    if days < 0:
        months -= 1
        pm = 12 if b.month == 1 else b.month - 1
        py = b.year - 1 if b.month == 1 else b.year
        days += calendar.monthrange(py, pm)[1]
    if months < 0:
        months += 12
        years -= 1
    return total, years, months, days


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression and return the exact result. Use this for ANY "
            "calculation — never do multi-digit or multi-step math yourself. Supports + - * / // % "
            "** parentheses and the functions sqrt, abs, round, floor, ceil, min, max, log, log10, "
            "exp, factorial, gcd, sin, cos, tan and constants pi, e. Turn a worded problem into an "
            "expression first, e.g. '15% of 240' -> '0.15*240', 'square root of 144' -> 'sqrt(144)', "
            "'twelve times seven' -> '12*7'. Don't use thousands separators (write 1234, not 1,234)."),
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description":
                           "The arithmetic expression to evaluate, e.g. '(1234*56)/7' or 'sqrt(144)+3'."},
        }, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "date_diff",
        "description": (
            "Exact number of days (plus a years/months/days breakdown) between two calendar dates. "
            "Use this for ANY 'how long since', 'how many days/weeks/months since', 'how long until', "
            "or 'how old' question — never count days across months yourself, you get it wrong. "
            "`start` is required (ISO YYYY-MM-DD); `end` defaults to today. Result is positive when "
            "start is in the past. Pull the date from what the user said or what you remember."),
        "parameters": {"type": "object", "properties": {
            "start": {"type": "string", "description": "The earlier date, ISO YYYY-MM-DD (e.g. a quit or birth date)."},
            "end": {"type": "string", "description": "The later date, ISO YYYY-MM-DD. Omit to use today."},
        }, "required": ["start"]}}},
]


class CalculatorProvider:
    name = "calculator"

    @property
    def enabled(self) -> bool:
        # Core capability, no external service - always on.
        return True

    def tool_specs(self) -> list[dict]:
        return TOOL_SPECS

    async def execute(self, tool: str, args: dict) -> dict:
        if tool == "calculate":
            return self._calculate(args)
        if tool == "date_diff":
            return self._date_diff(args)
        return {"error": f"unknown tool {tool!r}"}

    def _calculate(self, args: dict) -> dict:
        expr = (args.get("expression") or "").strip()
        if not expr:
            return {"error": "no expression given"}
        try:
            result = evaluate(expr)
        except ZeroDivisionError:
            return {"error": "division by zero"}
        except _Unsafe as e:
            return {"error": f"unsupported expression: {e}"}
        except (SyntaxError, ValueError, TypeError, OverflowError) as e:
            return {"error": f"could not evaluate {expr!r}: {e}"}
        if isinstance(result, float) and result.is_integer():
            result = int(result)  # 6.0 -> 6 reads better when spoken
        return {"expression": expr, "result": result}

    def _date_diff(self, args: dict) -> dict:
        try:
            start = _to_date(args.get("start"))
        except ValueError:
            return {"error": "start must be a date like 2026-01-27"}
        end_arg = args.get("end")
        try:
            end = _to_date(end_arg) if end_arg else datetime.now(timers._tz()).date()
        except ValueError:
            return {"error": "end must be a date like 2026-07-11"}
        total, y, mo, d = date_diff(start, end)
        n = abs(total)
        parts = []
        if y:
            parts.append(f"{y} year{'s' if y != 1 else ''}")
        if mo:
            parts.append(f"{mo} month{'s' if mo != 1 else ''}")
        if d or not parts:
            parts.append(f"{d} day{'s' if d != 1 else ''}")
        return {"start": start.isoformat(), "end": end.isoformat(), "days": n,
                "weeks": round(n / 7, 1), "breakdown": ", ".join(parts)}

    def system_note(self) -> str:
        return ""  # guidance lives in the 'math' skill's note (routed path)

    async def health(self) -> bool:
        return True
