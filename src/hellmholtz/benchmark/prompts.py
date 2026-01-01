"""
Collection of prompts for benchmarking LLMs across different categories.
"""

from hellmholtz.core.prompts import Message, Prompt

# Collection of prompts
PROMPTS: list[Prompt] = [
    Prompt(
        id="reasoning_001",
        category="reasoning",
        messages=[
            Message(
                role="user",
                content=(
                    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it "
                    "take 100 machines to make 100 widgets?"
                ),
            )
        ],
        description="Classic scaling problem",
        expected_output=None,
    ),
    Prompt(
        id="reasoning_002",
        category="reasoning",
        messages=[
            Message(
                role="user",
                content=(
                    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
                    "ball. How much does the ball cost?"
                ),
            )
        ],
        description="Price calculation puzzle",
        expected_output=None,
    ),
    Prompt(
        id="reasoning_003",
        category="reasoning",
        messages=[
            Message(
                role="user",
                content=(
                    "Solve this logic puzzle: There are three boxes, one contains only "
                    "apples, one contains only oranges, and one contains both apples and "
                    "oranges. The boxes have been incorrectly labeled such that no label "
                    "identifies the actual contents of the box it labels. Opening just one "
                    "box, and without looking in the box, you take out one piece of fruit. "
                    "By looking at the fruit, how can you immediately label all of the "
                    "boxes correctly?"
                ),
            )
        ],
        description="Logic puzzle with boxes and fruit",
        expected_output=None,
    ),
    Prompt(
        id="coding_001",
        category="coding",
        messages=[
            Message(
                role="user",
                content=(
                    "Write a Python function to calculate the Fibonacci sequence up to n "
                    "terms using a generator."
                ),
            )
        ],
        description="Python generator implementation",
        expected_output="Python code",
    ),
    Prompt(
        id="coding_002",
        category="coding",
        messages=[
            Message(
                role="user",
                content=(
                    "Implement a simple HTTP server in Go that responds with 'Hello, "
                    "World!' to any request."
                ),
            )
        ],
        description="Basic Go HTTP server",
        expected_output="Go code",
    ),
    Prompt(
        id="coding_003",
        category="coding",
        messages=[
            Message(
                role="user",
                content="Write a SQL query to find the second highest salary from an Employee "
                "table.",
            )
        ],
        description="SQL query for second highest value",
        expected_output="SQL query",
    ),
    Prompt(
        id="creative_001",
        category="creative",
        messages=[
            Message(
                role="user",
                content="Write a short poem about a robot discovering a flower in a wasteland.",
            )
        ],
        description="Poetry about AI and nature",
        expected_output="Poem",
    ),
    Prompt(
        id="creative_002",
        category="creative",
        messages=[
            Message(
                role="user",
                content="Write the opening paragraph of a mystery novel set in a futuristic "
                "underwater city.",
            )
        ],
        description="Creative writing opening",
        expected_output="Narrative paragraph",
    ),
    Prompt(
        id="creative_003",
        category="creative",
        messages=[
            Message(
                role="user",
                content="Compose a haiku about the feeling of compiling code successfully.",
            )
        ],
        description="Haiku about programming",
        expected_output="Haiku (5-7-5 syllables)",
    ),
    Prompt(
        id="knowledge_001",
        category="knowledge",
        messages=[
            Message(role="user", content="Explain the theory of relativity to a 5-year-old.")
        ],
        description="Simplifying complex science",
        expected_output="Simple explanation",
    ),
    Prompt(
        id="knowledge_002",
        category="knowledge",
        messages=[
            Message(role="user", content="What are the main differences between TCP and UDP?")
        ],
        description="Network protocol comparison",
        expected_output="Technical comparison",
    ),
    Prompt(
        id="knowledge_003",
        category="knowledge",
        messages=[
            Message(role="user", content="Summarize the key events of the French Revolution.")
        ],
        description="Historical summary",
        expected_output="Historical summary",
    ),
]


def get_all_prompts() -> list[Prompt]:
    """Return a flat list of all prompts."""
    return PROMPTS


def get_prompts_by_category(category: str) -> list[Prompt]:
    """Get all prompts for a specific category."""
    return [prompt for prompt in PROMPTS if prompt.category == category]


def get_prompt_by_id(prompt_id: str) -> Prompt | None:
    """Get a specific prompt by its ID."""
    for prompt in PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    return None
