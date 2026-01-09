# HeLLMholtz Agent Examples

This directory contains example scripts demonstrating the agent system with tool use capabilities.

## Prerequisites

Make sure you have HeLLMholtz installed and configured:

```bash
pip install -e .
```

Set up your API keys in `.env`:

```bash
OPENAI_API_KEY=your_key_here
# or
ANTHROPIC_API_KEY=your_key_here
```

## Running Examples

### Calculator Agent

Demonstrates using the agent with the CalculatorTool for mathematical problem solving:

```bash
python examples/agent_calculator.py
```

This example shows:
- Percentage calculations
- Geometric calculations (area of circle)
- Discount calculations
- Compound interest calculations

### File I/O Agent

Demonstrates using the agent with FileIOTool and CalculatorTool for data processing:

```bash
python examples/agent_files.py
```

This example shows:
- Reading data from files
- Processing and analyzing data
- Writing results to new files
- Listing directory contents

### Custom Tools

Demonstrates creating custom tools that extend the agent's capabilities:

```bash
python examples/custom_tool.py
```

This example shows:
- Creating a StringTool for text manipulation
- Creating a TemperatureTool for unit conversion
- Implementing the Tool interface
- Using custom tools with the agent

## Customizing Examples

All examples use `openai:gpt-4o` by default. You can modify the model in each script:

```python
config = AgentConfig(
    model="anthropic:claude-3-sonnet",  # Change to your preferred model
    max_iterations=10,
    temperature=0.1,
    verbose=True
)
```

Supported models:
- `openai:gpt-4o`
- `openai:gpt-4-turbo`
- `anthropic:claude-3-opus`
- `anthropic:claude-3-sonnet`
- `anthropic:claude-3-haiku`
- `google:gemini-pro`
- `blablador:<model-name>` (if configured)

## Troubleshooting

### "API key not found" errors

Make sure your `.env` file is properly configured with the necessary API keys for your chosen model provider.

### Import errors

Ensure HeLLMholtz is installed:

```bash
pip install -e .
```

### Web search not working

The SearchTool requires an optional dependency:

```bash
pip install duckduckgo-search
```

## More Information

For detailed documentation, see:
- [Agent Documentation](../docs/agent.md)
- [Main README](../README.md)
