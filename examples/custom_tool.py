"""
Example: Creating a custom tool for the agent.

This example demonstrates how to create a custom tool that the agent
can use, following the Tool interface.
"""

import logging
from typing import Any

from hellmholtz.agent import Agent, AgentConfig
from hellmholtz.agent.tools.base import Tool, ToolResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class StringTool(Tool):
    """Custom tool for string manipulation operations."""

    @property
    def name(self) -> str:
        return "string_tool"

    @property
    def description(self) -> str:
        return (
            "Performs string operations. "
            "Operations: 'reverse' (reverse a string), "
            "'uppercase' (convert to uppercase), "
            "'lowercase' (convert to lowercase), "
            "'length' (get string length). "
            "Input format: operation,text (e.g., 'reverse,hello')"
        )

    def execute(self, input_str: str, **kwargs: Any) -> ToolResult:
        """Execute string operation."""
        try:
            # Parse input
            if "," not in input_str:
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid input format. Use: operation,text",
                )

            operation, text = input_str.split(",", 1)
            operation = operation.strip().lower()
            text = text.strip()

            # Perform operation
            if operation == "reverse":
                result = text[::-1]
            elif operation == "uppercase":
                result = text.upper()
            elif operation == "lowercase":
                result = text.lower()
            elif operation == "length":
                result = str(len(text))
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}. "
                    "Use: reverse, uppercase, lowercase, or length",
                )

            return ToolResult(
                success=True,
                output=f"Result of {operation} on '{text}': {result}",
                error=None,
            )

        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"String operation failed: {str(e)}"
            )


class TemperatureTool(Tool):
    """Custom tool for temperature conversion."""

    @property
    def name(self) -> str:
        return "temperature"

    @property
    def description(self) -> str:
        return (
            "Converts temperatures between Celsius and Fahrenheit. "
            "Input format: 'value,from_unit,to_unit' "
            "(e.g., '32,F,C' or '100,C,F'). "
            "Units: C (Celsius), F (Fahrenheit)"
        )

    def execute(self, input_str: str, **kwargs: Any) -> ToolResult:
        """Execute temperature conversion."""
        try:
            # Parse input
            parts = input_str.split(",")
            if len(parts) != 3:
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid input. Use: value,from_unit,to_unit",
                )

            value_str, from_unit, to_unit = [p.strip() for p in parts]
            value = float(value_str)
            from_unit = from_unit.upper()
            to_unit = to_unit.upper()

            # Convert
            if from_unit == "C" and to_unit == "F":
                result = (value * 9 / 5) + 32
                unit_name = "Fahrenheit"
            elif from_unit == "F" and to_unit == "C":
                result = (value - 32) * 5 / 9
                unit_name = "Celsius"
            elif from_unit == to_unit:
                result = value
                unit_name = from_unit
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error="Invalid units. Use C (Celsius) or F (Fahrenheit)",
                )

            return ToolResult(
                success=True,
                output=f"{value}°{from_unit} = {result:.2f}°{to_unit} ({unit_name})",
                error=None,
            )

        except ValueError:
            return ToolResult(
                success=False, output="", error="Invalid temperature value"
            )
        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"Conversion failed: {str(e)}"
            )


def main() -> None:
    """Run agent with custom tools."""

    # Configure agent
    config = AgentConfig(
        model="openai:gpt-4o",  # Change to your preferred model
        max_iterations=8,
        temperature=0.1,
        verbose=True,
    )

    # Setup custom tools
    tools = [StringTool(), TemperatureTool()]

    # Create agent
    agent = Agent(config=config, tools=tools)

    print("\n" + "=" * 70)
    print("AGENT WITH CUSTOM TOOLS")
    print("=" * 70 + "\n")

    # Example tasks
    tasks = [
        "Reverse the string 'Hello World'",
        "Convert 100 degrees Celsius to Fahrenheit",
        "What is the length of 'HeLLMholtz'?",
        "Convert 32 degrees Fahrenheit to Celsius and tell me if that's freezing point",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}")
        print(f"Task {i}: {task}")
        print("=" * 70 + "\n")

        result = agent.run(task)

        if result.success:
            print(f"\n✓ Success! Answer: {result.answer}")
            print(f"Iterations used: {result.iterations}")
        else:
            print(f"\n✗ Failed: {result.answer}")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
