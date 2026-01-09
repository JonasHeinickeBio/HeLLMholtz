# Agent System with Tool Use

The HeLLMholtz agent system provides agentic capabilities using a ReAct (Reasoning + Acting) approach. The agent can break down complex tasks, reason about them, and use tools to gather information and perform actions.

## Overview

The agent system consists of:

- **Agent**: The main orchestrator that manages the reasoning loop
- **Tools**: Specialized utilities the agent can use (calculator, file I/O, web search)
- **ReAct Loop**: The core reasoning pattern that alternates between thinking and acting

## Architecture

### ReAct Pattern

The agent follows the ReAct (Reasoning + Acting) pattern:

1. **Thought**: Agent reasons about what to do next
2. **Action**: Agent decides to use a tool
3. **Action Input**: Agent provides input for the tool
4. **Observation**: Tool executes and returns a result
5. Repeat until the agent has enough information to provide a **Final Answer**

### Tool Interface

All tools implement a standard interface:

```python
from hellmholtz.agent.tools.base import Tool, ToolResult

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in agent commands."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does and how to use it."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
```

## Built-in Tools

### Calculator Tool

Performs safe mathematical calculations using Python's AST for expression evaluation.

**Supported Operations:**
- Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`)
- Power (`**`), Modulo (`%`), Floor Division (`//`)

**Example:**
```python
from hellmholtz.agent.tools import CalculatorTool

calc = CalculatorTool()
result = calc.execute("(15 * 3) + (20 / 4)")
# Result: 50.0
```

**Safety:** Uses AST parsing to prevent code execution, only allows mathematical operations.

### File I/O Tool

Provides safe file reading, writing, and directory listing capabilities.

**Operations:**
- `read`: Read file content
- `write`: Write content to file
- `list`: List directory contents

**Example:**
```python
from hellmholtz.agent.tools import FileIOTool

# With workspace restriction
file_io = FileIOTool(workspace_dir="/safe/workspace")

# Write file
result = file_io.execute(
    operation="write",
    path="/safe/workspace/test.txt",
    content="Hello, World!"
)

# Read file
result = file_io.execute(operation="read", path="/safe/workspace/test.txt")
```

**Safety:** 
- Workspace directory restriction prevents directory traversal
- Path validation ensures files stay within allowed directory
- UTF-8 encoding for text files

### Search Tool

Performs web searches using DuckDuckGo (no API key required).

**Example:**
```python
from hellmholtz.agent.tools import SearchTool

search = SearchTool(max_results=5)
result = search.execute("Python programming best practices")
```

**Requirements:**
```bash
pip install duckduckgo-search
```

**Note:** The search tool will return an error if `duckduckgo-search` is not installed, but other agent functionality will continue to work.

## Using the Agent

### Python API

```python
from hellmholtz.agent import Agent, AgentConfig, CalculatorTool, FileIOTool

# Configure agent
config = AgentConfig(
    model="openai:gpt-4o",
    max_iterations=10,
    temperature=0.1,
    verbose=True
)

# Setup tools
tools = [
    CalculatorTool(),
    FileIOTool(workspace_dir="/tmp/agent_workspace")
]

# Create and run agent
agent = Agent(config=config, tools=tools)
result = agent.run("What is 15% of 240?")

# Check result
if result.success:
    print(f"Answer: {result.answer}")
    print(f"Used {result.iterations} iterations")
else:
    print(f"Failed: {result.answer}")
```

### Command Line Interface

```bash
# Basic usage
hellm agent \
  --model openai:gpt-4o \
  "What is the square root of 144?"

# With verbose output to see reasoning process
hellm agent \
  --model openai:gpt-4o \
  --verbose \
  "Calculate the compound interest on $1000 at 5% for 3 years"

# With file operations (restricted to workspace)
hellm agent \
  --model anthropic:claude-3-sonnet \
  --workspace /tmp/my_workspace \
  "Create a file called notes.txt with a summary of Python's key features"

# With web search enabled
hellm agent \
  --model openai:gpt-4o \
  --enable-search \
  "What are the latest developments in quantum computing?"

# Custom iteration limit and temperature
hellm agent \
  --model openai:gpt-4o \
  --max-iterations 15 \
  --temperature 0.3 \
  "Solve this multi-step problem: First calculate 25^2, then divide by 5, then multiply by 3"
```

### CLI Options

- `--model`: LLM model to use (required)
- `--max-iterations`: Maximum reasoning iterations (default: 10)
- `--temperature`: LLM temperature (default: 0.1)
- `--verbose`: Show detailed reasoning process
- `--enable-search`: Enable web search tool
- `--workspace`: Workspace directory for file operations

## Agent Configuration

The `AgentConfig` class controls agent behavior:

```python
from hellmholtz.agent import AgentConfig

config = AgentConfig(
    model="openai:gpt-4o",      # Required: Model identifier
    max_iterations=10,           # Max reasoning steps (1-50)
    temperature=0.1,             # LLM temperature (0.0-2.0)
    verbose=False                # Print detailed logs
)
```

**Configuration Guidelines:**
- **max_iterations**: Higher values allow more complex reasoning but increase cost
- **temperature**: Lower (0.0-0.3) for precise tasks, higher (0.5-1.0) for creative tasks
- **verbose**: Enable for debugging or understanding agent reasoning

## Agent Result

The agent returns an `AgentResult` object:

```python
class AgentResult(BaseModel):
    success: bool                    # Whether agent completed successfully
    answer: str                      # Final answer or error message
    iterations: int                  # Number of iterations used
    thought_process: list[dict]      # Complete reasoning trace
```

**Example:**
```python
result = agent.run("What is 2 + 2?")

print(f"Success: {result.success}")
print(f"Answer: {result.answer}")
print(f"Iterations: {result.iterations}")

# Examine reasoning process
for step in result.thought_process:
    print(f"Iteration {step['iteration']}: {step['type']}")
    if step['type'] == 'action':
        print(f"  Action: {step['action']}")
        print(f"  Input: {step['action_input']}")
        print(f"  Observation: {step['observation']}")
```

## Creating Custom Tools

You can extend the agent with custom tools:

```python
from hellmholtz.agent.tools.base import Tool, ToolResult

class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Gets weather information for a location. "
            "Input: City name (e.g., 'London', 'New York'). "
            "Returns current weather conditions."
        )

    def execute(self, location: str, **kwargs) -> ToolResult:
        try:
            # Your weather API logic here
            weather_data = get_weather(location)
            return ToolResult(
                success=True,
                output=f"Weather in {location}: {weather_data}",
                error=None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to get weather: {str(e)}"
            )

# Use custom tool
agent = Agent(
    config=config,
    tools=[CalculatorTool(), WeatherTool()]
)
```

## Best Practices

### 1. Choose Appropriate Models

- **GPT-4**: Best for complex reasoning and multi-step tasks
- **GPT-3.5/Claude-3-Haiku**: Good for simpler tasks, faster and cheaper
- **Temperature**: Use 0.0-0.2 for precise tool use, higher for creative tasks

### 2. Set Reasonable Iteration Limits

```python
# Simple calculations: 3-5 iterations usually sufficient
config = AgentConfig(model="openai:gpt-4o", max_iterations=5)

# Complex multi-step reasoning: 10-20 iterations
config = AgentConfig(model="openai:gpt-4o", max_iterations=15)
```

### 3. Use Workspace for File Operations

Always specify a workspace directory to restrict file operations:

```python
file_io = FileIOTool(workspace_dir="/tmp/safe_workspace")
```

### 4. Enable Verbose Mode for Debugging

When developing or debugging, enable verbose mode:

```python
config = AgentConfig(model="openai:gpt-4o", verbose=True)
```

### 5. Handle Failures Gracefully

```python
result = agent.run(question)
if not result.success:
    logger.error(f"Agent failed: {result.answer}")
    # Implement fallback logic
```

## Example Use Cases

### Mathematical Problem Solving

```python
agent = Agent(
    config=AgentConfig(model="openai:gpt-4o", max_iterations=8),
    tools=[CalculatorTool()]
)

result = agent.run(
    "A store offers 20% discount on a $150 item. "
    "If there's also 8% sales tax, what's the final price?"
)
```

### Data Analysis with Files

```python
agent = Agent(
    config=AgentConfig(model="openai:gpt-4o", max_iterations=10),
    tools=[
        FileIOTool(workspace_dir="/tmp/data"),
        CalculatorTool()
    ]
)

result = agent.run(
    "Read the sales data from sales.csv, calculate the total, "
    "and write a summary to report.txt"
)
```

### Research with Web Search

```python
agent = Agent(
    config=AgentConfig(model="openai:gpt-4o", max_iterations=12),
    tools=[
        SearchTool(max_results=5),
        FileIOTool(workspace_dir="/tmp/research")
    ]
)

result = agent.run(
    "Research the top 3 programming languages in 2024 and "
    "create a summary file comparing their key features"
)
```

## Limitations and Considerations

### Token Limits

Long reasoning chains can consume significant tokens. Monitor usage and set appropriate `max_iterations`.

### Tool Execution Time

Some tools (especially web search) may be slow. Consider timeout mechanisms for production use.

### Error Recovery

The agent may not always recover from tool errors. Implement retry logic if needed:

```python
max_retries = 3
for attempt in range(max_retries):
    result = agent.run(question)
    if result.success:
        break
    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
```

### Model Capabilities

Not all models are equally good at tool use:
- **Recommended**: GPT-4, GPT-4-Turbo, Claude-3-Opus, Claude-3-Sonnet
- **May struggle**: Smaller models like GPT-3.5-turbo

## Troubleshooting

### Agent doesn't use tools

- Increase `max_iterations`
- Lower `temperature` (try 0.0-0.1)
- Ensure tool descriptions are clear
- Check if model supports tool use well

### "Maximum iterations reached"

- Increase `max_iterations` in config
- Simplify the question/task
- Break complex tasks into smaller steps

### Tool execution errors

- Check tool input format
- Verify workspace permissions (for FileIOTool)
- Ensure required libraries are installed (for SearchTool)

### Verbose output not showing

- Ensure `verbose=True` in `AgentConfig`
- Check logging level: `logging.basicConfig(level=logging.INFO)`

## API Reference

For detailed API documentation, see:

- [Agent API](../api/agent.md)
- [Tools API](../api/tools.md)

## Examples

More examples are available in the `examples/` directory:

- `examples/agent_calculator.py`: Mathematical problem solving
- `examples/agent_files.py`: File operations and data processing
- `examples/agent_research.py`: Web search and information gathering
- `examples/custom_tool.py`: Creating custom tools
