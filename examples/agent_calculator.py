"""
Example: Using the agent with calculator for mathematical problem solving.

This example demonstrates how to use the HeLLMholtz agent with the
CalculatorTool to solve multi-step mathematical problems.
"""

import logging

from hellmholtz.agent import Agent, AgentConfig, CalculatorTool

# Configure logging to see agent's thought process
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Run agent with calculator for mathematical problems."""

    # Configure agent
    config = AgentConfig(
        model="openai:gpt-4o",  # Change to your preferred model
        max_iterations=8,
        temperature=0.1,
        verbose=True,  # Show reasoning process
    )

    # Setup tools
    tools = [CalculatorTool()]

    # Create agent
    agent = Agent(config=config, tools=tools)

    # Example problems
    problems = [
        "What is 15% of 240?",
        "Calculate the area of a circle with radius 7. Use 3.14159 for pi.",
        "A store has a $120 item on sale for 25% off. What's the final price?",
        "What is the compound amount for $1000 invested at 5% annual interest "
        "for 3 years? Use the formula: A = P * (1 + r)^t",
    ]

    print("\n" + "=" * 70)
    print("AGENT CALCULATOR EXAMPLES")
    print("=" * 70 + "\n")

    for i, problem in enumerate(problems, 1):
        print(f"\n{'='*70}")
        print(f"Problem {i}: {problem}")
        print("=" * 70 + "\n")

        result = agent.run(problem)

        if result.success:
            print(f"\n✓ Success! Answer: {result.answer}")
            print(f"Iterations used: {result.iterations}")
        else:
            print(f"\n✗ Failed: {result.answer}")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
