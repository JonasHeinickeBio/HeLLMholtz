"""Tests for hellmholtz.core.prompts (Message, Prompt, load_prompts)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hellmholtz.core.prompts import Message, Prompt, load_prompts


# ── Message tests ─────────────────────────────────────────────────────────────


class TestMessage:
    def test_valid_message(self) -> None:
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert m.name is None

    def test_valid_with_name(self) -> None:
        m = Message(role="assistant", content="hi", name="bot")
        assert m.name == "bot"

    def test_system_role(self) -> None:
        m = Message(role="system", content="Be helpful")
        assert m.role == "system"

    def test_assistant_role(self) -> None:
        m = Message(role="assistant", content="OK")
        assert m.role == "assistant"

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="Role must be one of"):
            Message(role="tool", content="data")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError):
            Message(role="user", content="")

    def test_whitespace_content_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            Message(role="user", content="   ")

    def test_content_stripped(self) -> None:
        m = Message(role="user", content="  hello  ")
        assert m.content == "hello"

    def test_name_empty_string_becomes_none(self) -> None:
        m = Message(role="user", content="hi", name="")
        assert m.name is None

    def test_name_whitespace_becomes_none(self) -> None:
        m = Message(role="user", content="hi", name="  ")
        assert m.name is None


# ── Prompt tests ──────────────────────────────────────────────────────────────


class TestPrompt:
    def _make_prompt(self, **overrides) -> Prompt:
        defaults = {
            "id": "test-1",
            "category": "reasoning",
            "messages": [Message(role="user", content="What is 2+2?")],
        }
        defaults.update(overrides)
        return Prompt(**defaults)

    def test_valid_prompt(self) -> None:
        p = self._make_prompt()
        assert p.id == "test-1"
        assert p.category == "reasoning"
        assert len(p.messages) == 1

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="ID cannot be empty"):
            self._make_prompt(id="")

    def test_whitespace_id_raises(self) -> None:
        with pytest.raises(ValueError, match="ID cannot be empty"):
            self._make_prompt(id="   ")

    def test_empty_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Category cannot be empty"):
            self._make_prompt(category="")

    def test_whitespace_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Category cannot be empty"):
            self._make_prompt(category="  ")

    def test_no_user_message_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one user message"):
            self._make_prompt(messages=[
                Message(role="system", content="You are helpful"),
                Message(role="assistant", content="OK"),
            ])

    def test_multiple_user_messages_ok(self) -> None:
        p = self._make_prompt(messages=[
            Message(role="user", content="Q1"),
            Message(role="user", content="Q2"),
        ])
        assert len(p.messages) == 2


class TestPromptProperties:
    def _make_prompt(self) -> Prompt:
        return Prompt(
            id="p1",
            category="coding",
            messages=[
                Message(role="system", content="Be a coder"),
                Message(role="user", content="Write hello world"),
                Message(role="assistant", content="def hello(): ..."),
            ],
        )

    def test_user_message(self) -> None:
        p = self._make_prompt()
        assert p.user_message == "Write hello world"

    def test_system_message(self) -> None:
        p = self._make_prompt()
        assert p.system_message == "Be a coder"

    def test_no_system_message(self) -> None:
        p = Prompt(
            id="p1",
            category="coding",
            messages=[Message(role="user", content="hi")],
        )
        assert p.system_message is None

    def test_no_user_message_returns_empty(self) -> None:
        # user_message property returns "" when no user messages exist
        # We test the property by constructing a Prompt with only user messages
        # and checking that user_message works, then verify validation catches no-user
        p = Prompt(
            id="p1",
            category="coding",
            messages=[Message(role="user", content="hi")],
        )
        # Verify property works
        assert p.user_message == "hi"
        # Verify validation rejects missing user message
        with pytest.raises(ValueError, match="at least one user message"):
            Prompt(
                id="p1",
                category="coding",
                messages=[Message(role="system", content="sys")],
            )


class TestPromptSerialization:
    def _make_prompt(self) -> Prompt:
        return Prompt(
            id="p1",
            category="reasoning",
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="What is 3+3?"),
            ],
            description="A test prompt",
            expected_output="6",
        )

    def test_to_openai_format(self) -> None:
        p = self._make_prompt()
        fmt = p.to_openai_format()
        assert isinstance(fmt, list)
        assert len(fmt) == 2
        assert fmt[0]["role"] == "system"
        assert fmt[1]["role"] == "user"
        assert "name" not in fmt[0]

    def test_to_dict(self) -> None:
        p = self._make_prompt()
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "p1"
        assert d["category"] == "reasoning"
        assert len(d["messages"]) == 2

    def test_to_json(self) -> None:
        p = self._make_prompt()
        j = p.to_json()
        data = json.loads(j)
        assert data["id"] == "p1"
        assert len(data["messages"]) == 2

    def test_to_json_with_indent(self) -> None:
        p = self._make_prompt()
        j = p.to_json(indent=4)
        assert "    " in j

    def test_to_yaml(self) -> None:
        p = self._make_prompt()
        y = p.to_yaml()
        data = yaml.safe_load(y)
        assert data["id"] == "p1"
        assert data["category"] == "reasoning"


class TestPromptDeserialization:
    def _make_dict(self) -> dict:
        return {
            "id": "p1",
            "category": "coding",
            "messages": [{"role": "user", "content": "Write code"}],
            "description": "Test",
            "expected_output": "Code",
        }

    def test_from_dict(self) -> None:
        p = Prompt.from_dict(self._make_dict())
        assert p.id == "p1"
        assert p.category == "coding"
        assert p.messages[0].content == "Write code"

    def test_from_json(self) -> None:
        d = self._make_dict()
        j = json.dumps(d)
        p = Prompt.from_json(j)
        assert p.id == "p1"
        assert p.messages[0].role == "user"

    def test_from_yaml(self) -> None:
        d = self._make_dict()
        y = yaml.dump(d)
        p = Prompt.from_yaml(y)
        assert p.id == "p1"
        assert p.messages[0].content == "Write code"

    def test_from_dict_minimal(self) -> None:
        d = {"id": "x", "category": "test", "messages": [{"role": "user", "content": "hi"}]}
        p = Prompt.from_dict(d)
        assert p.description is None
        assert p.expected_output is None


# ── load_prompts tests ────────────────────────────────────────────────────────


class TestLoadPrompts:
    def test_json_list(self, tmp_path: Path) -> None:
        prompts_data = [
            {"id": "p1", "category": "a", "messages": [{"role": "user", "content": "Q1"}]},
            {"id": "p2", "category": "b", "messages": [{"role": "user", "content": "Q2"}]},
        ]
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(prompts_data))
        result = load_prompts(str(f))
        assert len(result) == 2
        assert result[0].id == "p1"

    def test_json_single_object(self, tmp_path: Path) -> None:
        prompt_data = {"id": "single", "category": "x", "messages": [{"role": "user", "content": "hello"}]}
        f = tmp_path / "single.json"
        f.write_text(json.dumps(prompt_data))
        result = load_prompts(str(f))
        assert len(result) == 1
        assert result[0].id == "single"

    def test_txt_format(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.txt"
        f.write_text("What is AI?\nExplain ML.\nWrite code.\n")
        result = load_prompts(str(f))
        assert len(result) == 3
        assert all(p.category == "custom" for p in result)
        assert all(p.messages[0].role == "user" for p in result)

    def test_txt_empty_lines_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("Q1\n\n\nQ2\n\n")
        result = load_prompts(str(f))
        assert len(result) == 2

    def test_category_filter(self, tmp_path: Path) -> None:
        prompts_data = [
            {"id": "p1", "category": "reasoning", "messages": [{"role": "user", "content": "Q1"}]},
            {"id": "p2", "category": "coding", "messages": [{"role": "user", "content": "Q2"}]},
        ]
        f = tmp_path / "filtered.json"
        f.write_text(json.dumps(prompts_data))
        result = load_prompts(str(f), category="reasoning")
        assert len(result) == 1
        assert result[0].category == "reasoning"

    def test_txt_with_category(self, tmp_path: Path) -> None:
        f = tmp_path / "tagged.txt"
        f.write_text("Q1\nQ2\n")
        result = load_prompts(str(f), category="math")
        assert all(p.category == "math" for p in result)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompts("/nonexistent/file.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(str(f))

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xml"
        f.write_text("<root/>")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_prompts(str(f))

    def test_format_override_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("Q1\nQ2\n")
        result = load_prompts(str(f), file_format="txt")
        assert len(result) == 2

    def test_format_override_json(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        prompts_data = [{"id": "p1", "category": "a", "messages": [{"role": "user", "content": "Q"}]}]
        f.write_text(json.dumps(prompts_data))
        result = load_prompts(str(f), file_format="json")
        assert len(result) == 1

    def test_invalid_format_override_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_prompts(str(f), file_format="csv")

    def test_json_empty_array(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.json"
        f.write_text("[]")
        result = load_prompts(str(f))
        assert len(result) == 0

    def test_txt_with_md_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "prompts.md"
        f.write_text("Q1\nQ2\n")
        result = load_prompts(str(f))
        assert len(result) == 2
