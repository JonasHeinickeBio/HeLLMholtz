"""
Example: Using the agent with file I/O for data processing.

This example demonstrates how to use the HeLLMholtz agent with the
FileIOTool to read, process, and write files.
"""

import logging
import tempfile
from pathlib import Path

from hellmholtz.agent import Agent, AgentConfig, CalculatorTool, FileIOTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Run agent with file I/O capabilities."""

    # Create a temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create sample data file
        data_file = workspace / "sales_data.txt"
        data_file.write_text(
            """January: 1200
February: 1500
March: 1350
April: 1800"""
        )

        # Configure agent
        config = AgentConfig(
            model="openai:gpt-4o",  # Change to your preferred model
            max_iterations=15,
            temperature=0.1,
            verbose=True,
        )

        # Setup tools with workspace restriction
        tools = [
            CalculatorTool(),
            FileIOTool(workspace_dir=str(workspace)),
        ]

        # Create agent
        agent = Agent(config=config, tools=tools)

        print("\n" + "=" * 70)
        print("AGENT FILE I/O EXAMPLES")
        print("=" * 70 + "\n")

        # Task 1: Read and analyze data
        print("\nTask 1: Analyze sales data")
        print("-" * 70)

        result = agent.run(
            f"Read the file {data_file.name}, calculate the total sales "
            f"and average sales, then write a summary to summary.txt"
        )

        if result.success:
            print(f"\n✓ Success! {result.answer}")

            # Show created file
            summary_file = workspace / "summary.txt"
            if summary_file.exists():
                print(f"\nCreated file contents:")
                print(summary_file.read_text())
        else:
            print(f"\n✗ Failed: {result.answer}")

        # Task 2: List files
        print("\n\nTask 2: List workspace files")
        print("-" * 70)

        result = agent.run(f"List all files in the directory and describe what you find")

        if result.success:
            print(f"\n✓ Success! {result.answer}")
        else:
            print(f"\n✗ Failed: {result.answer}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
