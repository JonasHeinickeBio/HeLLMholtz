"""
Tests for agent functionality.

This module contains tests for the ReAct agent implementation,
including agent configuration, execution, and tool integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from hellmholtz.agent import Agent, AgentConfig, AgentResult
from hellmholtz.agent.tools import CalculatorTool, Tool, ToolResult


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock", description: str = "Mock tool") -> None:
        self._name = name
        self._description = description
        self.call_count = 0
        self.last_input: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, *args: str, **kwargs: str) -> ToolResult:
        self.call_count += 1
        self.last_input = args[0] if args else None
        return ToolResult(
            success=True, output=f"Mock result for: {self.last_input}", error=None
        )


class TestAgentConfig:
    """Test suite for AgentConfig."""

    def test_agent_config_defaults(self) -> None:
        """Test AgentConfig with default values."""
        config = AgentConfig(model="openai:gpt-4o")
        assert config.model == "openai:gpt-4o"
        assert config.max_iterations == 10
        assert config.temperature == 0.1
        assert config.verbose is False

    def test_agent_config_custom_values(self) -> None:
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            model="anthropic:claude-3-sonnet",
            max_iterations=20,
            temperature=0.5,
            verbose=True,
        )
        assert config.model == "anthropic:claude-3-sonnet"
        assert config.max_iterations == 20
        assert config.temperature == 0.5
        assert config.verbose is True

    def test_agent_config_validation(self) -> None:
        """Test AgentConfig validation."""
        # Test invalid max_iterations
        with pytest.raises(Exception):  # Pydantic validation error
            AgentConfig(model="openai:gpt-4o", max_iterations=0)

        with pytest.raises(Exception):
            AgentConfig(model="openai:gpt-4o", max_iterations=100)

        # Test invalid temperature
        with pytest.raises(Exception):
            AgentConfig(model="openai:gpt-4o", temperature=-1)

        with pytest.raises(Exception):
            AgentConfig(model="openai:gpt-4o", temperature=3.0)


class TestAgent:
    """Test suite for Agent class."""

    @pytest.fixture
    def basic_config(self) -> AgentConfig:
        """Create basic agent config."""
        return AgentConfig(model="openai:gpt-4o", max_iterations=3, verbose=False)

    @pytest.fixture
    def mock_tool(self) -> MockTool:
        """Create mock tool."""
        return MockTool()

    def test_agent_initialization(self, basic_config: AgentConfig) -> None:
        """Test agent initialization."""
        tools = [CalculatorTool()]
        agent = Agent(config=basic_config, tools=tools)

        assert agent.config == basic_config
        assert len(agent.tools) == 1
        assert "calculator" in agent.tools

    def test_get_tools_description(self, basic_config: AgentConfig) -> None:
        """Test tools description formatting."""
        tools = [CalculatorTool(), MockTool(name="test", description="Test tool")]
        agent = Agent(config=basic_config, tools=tools)

        description = agent._get_tools_description()
        assert "calculator" in description
        assert "test" in description

    def test_parse_action_valid(self, basic_config: AgentConfig, mock_tool: MockTool) -> None:
        """Test parsing valid action from agent response."""
        agent = Agent(config=basic_config, tools=[mock_tool])

        text = """
        Thought: I need to use the mock tool
        Action: mock
        Action Input: test input
        """

        action, action_input = agent._parse_action(text)
        assert action == "mock"
        assert action_input == "test input"

    def test_parse_action_invalid(
        self, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test parsing invalid action format."""
        agent = Agent(config=basic_config, tools=[mock_tool])

        text = "Just a thought without action"

        action, action_input = agent._parse_action(text)
        assert action is None
        assert action_input is None

    def test_execute_tool_success(
        self, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test successful tool execution."""
        agent = Agent(config=basic_config, tools=[mock_tool])

        result = agent._execute_tool("mock", "test input")
        assert "Mock result" in result
        assert mock_tool.call_count == 1
        assert mock_tool.last_input == "test input"

    def test_execute_tool_unknown(
        self, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test execution of unknown tool."""
        agent = Agent(config=basic_config, tools=[mock_tool])

        result = agent._execute_tool("unknown_tool", "input")
        assert "Unknown tool" in result
        assert mock_tool.call_count == 0

    def test_execute_tool_error_handling(self, basic_config: AgentConfig) -> None:
        """Test tool execution error handling."""

        class ErrorTool(Tool):
            @property
            def name(self) -> str:
                return "error_tool"

            @property
            def description(self) -> str:
                return "Tool that raises error"

            def execute(self, *args: str, **kwargs: str) -> ToolResult:
                raise ValueError("Test error")

        agent = Agent(config=basic_config, tools=[ErrorTool()])
        result = agent._execute_tool("error_tool", "input")
        assert "Error executing" in result

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_run_with_final_answer(
        self, mock_chat: MagicMock, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test agent run with immediate final answer."""
        mock_chat.return_value = (
            "Thought: I can answer this directly\n" "Final Answer: 42"
        )

        agent = Agent(config=basic_config, tools=[mock_tool])
        result = agent.run("What is the answer?")

        assert result.success is True
        assert "42" in result.answer
        assert result.iterations == 1
        assert mock_tool.call_count == 0

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_run_with_tool_use(
        self, mock_chat: MagicMock, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test agent run with tool use."""
        # First response uses tool, second provides final answer
        mock_chat.side_effect = [
            "Thought: Need to use mock tool\nAction: mock\nAction Input: test",
            "Thought: Got the result\nFinal Answer: Tool returned a result",
        ]

        agent = Agent(config=basic_config, tools=[mock_tool])
        result = agent.run("Test question")

        assert result.success is True
        assert result.iterations == 2
        assert mock_tool.call_count == 1
        assert len(result.thought_process) == 2

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_max_iterations(
        self, mock_chat: MagicMock, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test agent respects max iterations."""
        # Always return action without final answer
        mock_chat.return_value = "Thought: Keep going\nAction: mock\nAction Input: test"

        agent = Agent(config=basic_config, tools=[mock_tool])
        result = agent.run("Test question")

        assert result.success is False
        assert result.iterations == basic_config.max_iterations
        assert "Maximum iterations" in result.answer

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_handles_invalid_format(
        self, mock_chat: MagicMock, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test agent handles invalid response format."""
        # First response is invalid, second provides final answer
        mock_chat.side_effect = [
            "Just some random text",
            "Thought: Now I understand\nFinal Answer: Done",
        ]

        agent = Agent(config=basic_config, tools=[mock_tool])
        result = agent.run("Test question")

        assert result.success is True
        assert result.iterations == 2

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_error_handling(
        self, mock_chat: MagicMock, basic_config: AgentConfig, mock_tool: MockTool
    ) -> None:
        """Test agent handles errors during execution."""
        mock_chat.side_effect = Exception("LLM API error")

        agent = Agent(config=basic_config, tools=[mock_tool])
        result = agent.run("Test question")

        assert result.success is False
        assert "Error during execution" in result.answer


class TestAgentResult:
    """Test suite for AgentResult."""

    def test_agent_result_success(self) -> None:
        """Test successful agent result."""
        result = AgentResult(
            success=True,
            answer="The answer is 42",
            iterations=3,
            thought_process=[{"iteration": 1, "type": "action"}],
        )
        assert result.success is True
        assert result.answer == "The answer is 42"
        assert result.iterations == 3
        assert len(result.thought_process) == 1

    def test_agent_result_failure(self) -> None:
        """Test failed agent result."""
        result = AgentResult(
            success=False,
            answer="Maximum iterations reached",
            iterations=10,
            thought_process=[],
        )
        assert result.success is False
        assert "Maximum iterations" in result.answer


class TestAgentIntegration:
    """Integration tests for agent with real tools."""

    @pytest.fixture
    def calculator_config(self) -> AgentConfig:
        """Create config for calculator tests."""
        return AgentConfig(model="openai:gpt-4o", max_iterations=5, verbose=False)

    @patch("hellmholtz.agent.agent.chat")
    def test_agent_with_calculator(
        self, mock_chat: MagicMock, calculator_config: AgentConfig
    ) -> None:
        """Test agent using calculator tool."""
        # Simulate agent using calculator
        mock_chat.side_effect = [
            "Thought: I need to calculate\nAction: calculator\nAction Input: 2 + 2",
            "Thought: Got the result\nFinal Answer: The answer is 4",
        ]

        agent = Agent(config=calculator_config, tools=[CalculatorTool()])
        result = agent.run("What is 2 + 2?")

        assert result.success is True
        assert "4" in result.answer
        # Check that calculator was actually used
        assert any(
            step.get("action") == "calculator" for step in result.thought_process
        )
