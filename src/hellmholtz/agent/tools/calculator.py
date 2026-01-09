"""Calculator tool for mathematical operations."""

import ast
import operator
from typing import Any

from hellmholtz.agent.tools.base import Tool, ToolResult


class CalculatorTool(Tool):
    """Tool for performing safe mathematical calculations.

    Uses Python's AST to safely evaluate mathematical expressions
    without executing arbitrary code.
    """

    # Allowed operators for safe evaluation
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def name(self) -> str:
        """Tool name."""
        return "calculator"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Performs mathematical calculations. "
            "Input: A mathematical expression as a string (e.g., '2 + 2', '(10 * 5) / 2'). "
            "Supports +, -, *, /, **, %, // operations."
        )

    def _eval_expr(self, node: ast.expr) -> float:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return float(node.value)
        elif isinstance(node, ast.Num):  # Backwards compatibility
            return float(node.n)
        elif isinstance(node, ast.BinOp):
            op_func = self._OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_func = self._OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            operand = self._eval_expr(node.operand)
            return op_func(operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        """Execute mathematical calculation.

        Args:
            expression: Mathematical expression to evaluate
            **kwargs: Additional arguments (ignored)

        Returns:
            ToolResult with calculation result or error
        """
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode="eval")

            # Evaluate expression
            result = self._eval_expr(tree.body)

            return ToolResult(
                success=True, output=f"Result: {result}", error=None
            )
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                output="",
                error="Division by zero",
            )
        except (ValueError, SyntaxError, TypeError) as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid expression: {str(e)}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation error: {str(e)}",
            )
