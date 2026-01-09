"""ReAct agent implementation with tool use."""

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from hellmholtz.agent.tools.base import Tool
from hellmholtz.client import chat

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for ReAct agent."""

    model: str = Field(..., description="LLM model to use (e.g., openai:gpt-4o)")
    max_iterations: int = Field(
        default=10, description="Maximum reasoning iterations", ge=1, le=50
    )
    temperature: float = Field(default=0.1, description="LLM temperature", ge=0.0, le=2.0)
    verbose: bool = Field(default=False, description="Enable verbose logging")


class AgentResult(BaseModel):
    """Result from agent execution."""

    success: bool = Field(..., description="Whether agent completed successfully")
    answer: str = Field(..., description="Final answer or error message")
    iterations: int = Field(..., description="Number of iterations used")
    thought_process: list[dict[str, Any]] = Field(
        default_factory=list, description="Complete thought process"
    )


class Agent:
    """ReAct agent that uses tools to solve complex tasks.

    Implements the ReAct (Reasoning + Acting) paradigm where the agent
    alternates between thinking about the task and using tools.
    """

    REACT_PROMPT = """You are a helpful AI assistant that can use tools to answer questions and solve tasks.

Available tools:
{tools}

Use the following format for each step:

Thought: Think about what to do next
Action: tool_name
Action Input: input for the tool
Observation: [Tool result will be inserted here]

After enough observations, provide the final answer:

Thought: I now have enough information to answer
Final Answer: [Your answer here]

Important:
- Use tools when you need information or calculations
- One tool per Action
- Parse Action Input carefully - it should match what the tool expects
- After receiving an Observation, think about what it means
- Provide Final Answer only when you have sufficient information

Question: {question}

Begin!
"""

    def __init__(self, config: AgentConfig, tools: list[Tool]) -> None:
        """Initialize agent with configuration and tools.

        Args:
            config: Agent configuration
            tools: List of available tools
        """
        self.config = config
        self.tools = {tool.name: tool for tool in tools}
        self.thought_process: list[dict[str, Any]] = []

    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self.tools.values())

    def _parse_action(self, text: str) -> tuple[str | None, str | None]:
        """Parse action and action input from agent response.

        Args:
            text: Agent response text

        Returns:
            Tuple of (action_name, action_input) or (None, None) if not found
        """
        # Look for Action: and Action Input: patterns
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)

        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return action, action_input

        return None, None

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return its result.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool

        Returns:
            Tool execution result as string
        """
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"

        tool = self.tools[tool_name]

        try:
            # Try to parse JSON input for structured tools
            if tool_name == "file_io":
                try:
                    kwargs = json.loads(tool_input)
                    result = tool.execute(**kwargs)
                except json.JSONDecodeError:
                    result = tool.execute(operation="read", path=tool_input)
            else:
                # Simple string input for other tools
                result = tool.execute(tool_input)

            return str(result)

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    def run(self, question: str) -> AgentResult:
        """Run the agent on a question.

        Args:
            question: Question or task to solve

        Returns:
            AgentResult with answer and execution details
        """
        self.thought_process = []

        # Build initial prompt
        prompt = self.REACT_PROMPT.format(
            tools=self._get_tools_description(), question=question
        )

        conversation = [{"role": "user", "content": prompt}]

        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                logger.info(f"Iteration {iteration}/{self.config.max_iterations}")

            try:
                # Get agent response
                response = chat(
                    model=self.config.model,
                    messages=conversation,
                    temperature=self.config.temperature,
                )

                if self.config.verbose:
                    logger.info(f"Agent response:\n{response}")

                # Check for final answer
                if "Final Answer:" in response:
                    final_answer_match = re.search(
                        r"Final Answer:\s*(.+)", response, re.IGNORECASE | re.DOTALL
                    )
                    if final_answer_match:
                        answer = final_answer_match.group(1).strip()
                        self.thought_process.append(
                            {
                                "iteration": iteration,
                                "type": "final_answer",
                                "content": response,
                                "answer": answer,
                            }
                        )
                        return AgentResult(
                            success=True,
                            answer=answer,
                            iterations=iteration,
                            thought_process=self.thought_process,
                        )

                # Parse action
                action, action_input = self._parse_action(response)

                if action is None:
                    # Agent didn't provide valid action, prompt it
                    self.thought_process.append(
                        {
                            "iteration": iteration,
                            "type": "invalid_format",
                            "content": response,
                        }
                    )
                    conversation.append({"role": "assistant", "content": response})
                    conversation.append(
                        {
                            "role": "user",
                            "content": "Please provide a valid Action and Action Input, or a Final Answer.",
                        }
                    )
                    continue

                # Execute tool
                observation = self._execute_tool(action, action_input)

                self.thought_process.append(
                    {
                        "iteration": iteration,
                        "type": "action",
                        "thought": response,
                        "action": action,
                        "action_input": action_input,
                        "observation": observation,
                    }
                )

                if self.config.verbose:
                    logger.info(f"Observation:\n{observation}")

                # Add to conversation
                conversation.append({"role": "assistant", "content": response})
                conversation.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                return AgentResult(
                    success=False,
                    answer=f"Error during execution: {str(e)}",
                    iterations=iteration,
                    thought_process=self.thought_process,
                )

        # Max iterations reached
        return AgentResult(
            success=False,
            answer=f"Maximum iterations ({self.config.max_iterations}) reached without final answer",
            iterations=self.config.max_iterations,
            thought_process=self.thought_process,
        )
