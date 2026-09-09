"""Unit tests for the calculator tool (#54).

Covers the safe evaluator (correctness + rejection of anything non-arithmetic),
the provider's execute() contract, and the math routing heuristic. Stdlib only.

Run from ``backend/``:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import tempfile
import unittest

os.environ["NOVA_VOICE_HISTORY_DB"] = os.path.join(tempfile.mkdtemp(), "test_calc.db")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date  # noqa: E402

from app import skills  # noqa: E402
from app.plugins.calculator import CalculatorProvider, evaluate, date_diff, _Unsafe  # noqa: E402


class Evaluator(unittest.TestCase):
    def test_arithmetic_is_exact(self):
        self.assertEqual(evaluate("1234*56"), 69104)
        self.assertEqual(evaluate("(1234*56)/7"), 9872.0)
        self.assertEqual(evaluate("0.15*240"), 36.0)
        self.assertEqual(evaluate("2**10"), 1024)
        self.assertEqual(evaluate("17 % 5"), 2)
        self.assertEqual(evaluate("-3 + 4"), 1)

    def test_functions_and_constants(self):
        self.assertEqual(evaluate("sqrt(144)"), 12.0)
        self.assertEqual(evaluate("max(3, 9, 5)"), 9)
        self.assertEqual(evaluate("factorial(5)"), 120)
        self.assertAlmostEqual(evaluate("pi"), 3.14159, places=4)

    def test_rejects_non_arithmetic(self):
        for expr in ("__import__('os')", "open('x')", "x + 1", "lambda: 1", "[1,2]"):
            with self.assertRaises((_Unsafe, SyntaxError, ValueError), msg=expr):
                evaluate(expr)

    def test_guards_blowups(self):
        with self.assertRaises(_Unsafe):
            evaluate("10**99999")
        with self.assertRaises(_Unsafe):
            evaluate("factorial(100000)")


class DateDiff(unittest.TestCase):
    def test_the_smoking_pot_case(self):
        # The exact scenario the model kept getting wrong (said 177/213/24; it's 165).
        total, y, mo, d = date_diff(date(2026, 1, 27), date(2026, 7, 11))
        self.assertEqual(total, 165)
        self.assertEqual((y, mo, d), (0, 5, 14))  # 5 months, 14 days

    def test_calendar_correct_across_month_lengths(self):
        self.assertEqual(date_diff(date(2026, 1, 31), date(2026, 3, 1))[0], 29)   # Feb 2026 = 28d
        self.assertEqual(date_diff(date(2024, 2, 29), date(2025, 2, 28))[0], 365)  # leap span
        self.assertEqual(date_diff(date(2026, 7, 11), date(2026, 7, 11))[0], 0)

    def test_negative_when_end_before_start(self):
        self.assertEqual(date_diff(date(2026, 7, 11), date(2026, 1, 27))[0], -165)

    def test_provider_defaults_end_to_today_and_shapes(self):
        p = CalculatorProvider()
        out = asyncio.run(p.execute("date_diff", {"start": "2026-01-27", "end": "2026-07-11"}))
        self.assertEqual(out["days"], 165)
        self.assertEqual(out["breakdown"], "5 months, 14 days")
        # end omitted → uses today (just assert it computes without error)
        self.assertIn("days", asyncio.run(p.execute("date_diff", {"start": "2026-01-27"})))
        self.assertIn("error", asyncio.run(p.execute("date_diff", {"start": "not a date"})))


class ProviderContract(unittest.TestCase):
    def setUp(self):
        self.p = CalculatorProvider()

    def run_tool(self, args):
        return asyncio.run(self.p.execute("calculate", args))

    def test_success_shape_and_int_tidy(self):
        out = self.run_tool({"expression": "(1234*56)/7"})
        self.assertEqual(out["result"], 9872)          # 9872.0 tidied to int
        self.assertEqual(out["expression"], "(1234*56)/7")

    def test_division_by_zero(self):
        self.assertIn("error", self.run_tool({"expression": "5/0"}))

    def test_garbage_returns_error_not_raise(self):
        self.assertIn("error", self.run_tool({"expression": "15% of 240"}))  # English, not an expr
        self.assertIn("error", self.run_tool({"expression": ""}))

    def test_unknown_tool(self):
        self.assertIn("error", asyncio.run(self.p.execute("bogus", {})))


class MathRouting(unittest.TestCase):
    def test_looks_like_math_true(self):
        for m in ("what is 1234 times 56", "12*7", "15 percent of 240",
                  "square root of 144", "divide 100 by 4", "5 plus 3", "2 ** 8"):
            self.assertTrue(skills.looks_like_math(m), m)

    def test_looks_like_math_false(self):
        for m in ("what time is it", "turn on the lights at 5", "remind me in 10 minutes",
                  "what's on 2026-07-11", "add milk to my list"):
            self.assertFalse(skills.looks_like_math(m), m)

    def test_math_skill_is_registered(self):
        self.assertIn("math", {s.name for s in skills.SKILLS})
        self.assertIn("math", skills._BY_NAME)


if __name__ == "__main__":
    unittest.main()
