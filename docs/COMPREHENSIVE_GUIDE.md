# Comprehensive HeLLMholtz & OpenCode Manual for Colleagues

This guide provides a complete overview of the **HeLLMholtz LLM Suite**, **OpenCode**, and their integration with **Helmholtz Blablador**. Use this as your single reference for LLM operations at Helmholtz.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is HeLLMholtz?](#what-is-hellmholtz)
3. [What is OpenCode?](#what-is-opencode)
4. [Installation & Setup](#installation--setup)
   - [System Dependencies](#system-dependencies)
   - [Python Package Managers](#python-package-managers)
   - [HeLLMholtz Installation](#hellmholtz-installation)
   - [OpenCode Configuration](#opencode-configuration)
5. [Core Concepts](#core-concepts)
6. [HeLLMholtz Usage Guide](#hellmholtz-usage-guide)
7. [OpenCode Usage Guide](#opencode-usage-guide)
8. [Blablador Integration](#blablador-integration)
9. [Agent System](#agent-system)
10. [Advanced Features](#advanced-features)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### For HeLLMholtz (Command Line)
```bash
# Install
pip install "hellmholtz[eval,proxy]"

# Configure once
hellm-setup --api-key "YOUR_BLABLADOR_TOKEN" \
            --base-url "https://api.blablador.fz-juelich.de/v1"

# Chat with a model
hellm chat --model blablador:alias-code "Explain quantum computing"

# List available models
hellm models

# Run benchmarks
hellm bench --models blablador:alias-code,blablador:alias-fast \
            --prompts-category reasoning
```

### For OpenCode (AI Coding Assistant)
```bash
# Install OpenCode from source
curl -fsSL https://opencode.ai/install | bash

# Check if OpenCode is configured via ~/.config/opencode/opencode.json
# Already configured for Blablador with models: alias-code, alias-fast, alias-large, alias-huge

# Run OpenCode
opencode

# Use in terminal
opencode "Write a Python function to calculate fibonacci numbers"
```

---

## What is OpenCode?

**OpenCode** is an open-source CLI AI coding assistant (like Claude Code, but private and open-source). It's configured to use **Blablador models** as its brain.

### Key Features

- **Private AI Assistant**: All processing happens with your Blablador API key
- **Model Switching**: Can use different Blablador models for different tasks
- **Context Pruning**: Automatic DCP (Dynamic Context Pruning) for efficiency
- **Obsidian Integration**: Connects to your coding-brain vault for knowledge management
- **MCP Server Support**: Supports many Model Context Protocol servers

### Blablador Models in OpenCode

| Model Alias | Purpose | Context | Max Output |
|-------------|---------|---------|------------|
| `alias-code` | **Primary** coding assistant | 98K | 32K |
| `alias-fast` | Fast responses, quick tasks | 98K | 32K |
| `alias-large` | Large context handling | 98K | 32K |
| `alias-huge` | Maximum capability (Minimax) | 98K | 32K |

---

## Installation & Setup

### System Dependencies

Install required system packages first:

```bash
# Install pipx (Python package manager for CLI tools)
pip install pipx

# Install poetry (dependency manager)
pipx install poetry
```

**What are pipx and poetry?**

- **pipx**: Installs Python CLI tools in isolated environments (no conflicts)
- **poetry**: Manages project dependencies and virtual environments

### Python Package Managers

| Tool | Purpose | Use Case |
|------|---------|----------|
| `pip` | Install Python packages | Installing libraries in current environment |
| `pipx` | Install CLI tools | Installing applications like `hellm`, `black`, `ruff` |
| `poetry` | Manage project deps | Creating reproducible Python projects |

---

### Getting Your Blablador Token

1. Visit [codebase.helmholtz.cloud](https://codebase.helmholtz.cloud)
2. Sign in with your Helmholtz ID
3. Go to **User Settings** → **Access** → **Personal Access Tokens**
4. Click **New Token**
5. Configure token:
   - **Name**: `Blablador API Token`
   - **Scope**: Select `read_user`
   - **Expiration**: Choose appropriate duration
6. Click **Create personal access token**
7. Copy the token (shown only once!)
8. Use this token as your API key

#### User-Level Configuration (Recommended)
```bash
# Configure once (works globally for all projects)
hellm-setup --api-key "YOUR_BLABLADOR_TOKEN" \
            --base-url "https://api.blablador.fz-juelich.de/v1" \
            --default-models "alias-code,alias-fast"
```

Configuration is stored in: `~/.config/hellmholtz/.env`

#### Project-Level Configuration
```bash
# In your project directory
cp .env.example .env
# Edit .env with your API keys
```

### OpenCode Configuration

OpenCode is already configured at `~/.config/opencode/opencode.json` with:

- **Primary model**: `blablador/alias-code`
- **Small model**: `blablador/alias-fast`
- **MCP servers**: Obsidian, Markitdown, Docling, Semantic Scholar, Europe PMC
- **Plugin**: Obsidian integration for knowledge management

---

### Provider Format

All HeLLMholtz commands accept models in the format: `provider:model-name`

**Examples:**
- `blablador:alias-code` - Use Blablador's code-optimized model
- `blablador:alias-fast` - Use Blablador's fast-response model
- `openai:gpt-4o` - Use OpenAI's GPT-4o
- `anthropic:claude-3-opus` - Use Anthropic's Claude 3 Opus

### Model Names in Blablador

Blablador models are accessed via **aliases**:

| Alias | Description | Use Case |
|-------|-------------|----------|
| `alias-code` | Code-optimized model | Programming tasks |
| `alias-fast` | Fast responses | Quick questions |
| `alias-large` | Large context | Long documents |
| `alias-huge` | Max capability | Complex reasoning |


## HeLLMholtz Usage Guide

### Chat Interface

#### Basic Chat
```bash
# Simple chat
hellm chat --model blablador:alias-code "Explain quantum computing"

# With system prompt
hellm chat --model openai:gpt-4o --system "You are a Python expert" "How do I use decorators?"

# Interactive mode
hellm chat --model blablador:alias-code --interactive
```

#### Python API
```python
from hellmholtz.client import chat

# Simple chat
response = chat("blablador:alias-code", "Hello, how are you?")
print(response)

# With conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]
response = chat("anthropic:claude-3-opus", messages)
```

### Model Management

#### List Available Models
```bash
# List all Blablador models
hellm models

# Get model details (token limits, etc.)
hellm models --details
```

#### Monitor Model Health
```bash
# Basic monitoring
hellm monitor

# Test actual accessibility
hellm monitor --test-accessibility

# Check configuration consistency
hellm monitor --check-config
```

### Benchmarking

#### Using Built-in Prompt Categories
```bash
# Reasoning prompts (default)
hellm bench --models blablador:alias-code,openai:gpt-4o --prompts-category reasoning

# All categories
hellm bench --models blablador:alias-code --all-prompts

# Available categories: reasoning, coding, creative, knowledge
```

#### Using Custom Prompts File

**Text format** (one prompt per line):
```bash
hellm bench --models blablador:alias-code --prompts-file prompts.txt
```

**JSON format** (structured with metadata):
```json
[
  {
    "id": "custom_reasoning_001",
    "category": "reasoning",
    "messages": [
      {"role": "user", "content": "Your prompt here"}
    ],
    "description": "What this tests",
    "expected_output": "Expected answer"
  }
]
```

#### Advanced Benchmarking
```bash
hellm bench \
  --models blablador:alias-code,blablador:alias-fast,openai:gpt-4o \
  --prompts-file custom_prompts.json \
  --prompts-category reasoning \
  --temperatures 0.1,0.7,1.0 \
  --replications 3 \
  --evaluate-with openai:gpt-4o \
  --results-dir results/
```

**Parameters:**
- `--temperatures`: Test different creativity levels (0.1 = focused, 1.0 = creative)
- `--replications`: Number of times to run each test (for reliability)
- `--evaluate-with`: Use an LLM to score responses (LLM-as-a-Judge)

### Throughput Benchmarking

Measure tokens per second for performance analysis:
```bash
hellm bench-throughput --model blablador:alias-code --requests 100 --concurrency 10
```

### Reporting & Analysis

#### Generate Reports
```bash
# Markdown report
hellm report --input-file results/benchmark_20241214_120000.json --output-file report.md

# HTML report with interactive charts
hellm analyze results/benchmark_latest.json --html-report analysis_report.html
```

#### Python Analysis
```python
from hellmholtz.evaluation_analysis import EvaluationAnalyzer

analyzer = EvaluationAnalyzer()
analysis = analyzer.analyze_evaluation_results("results/benchmark_latest.json")
analyzer.print_analysis_summary(analysis)
```

### Model Discovery & Configuration

Check available Blablador models and their token limits:
```python
from hellmholtz.providers.blablador_config import get_token_limit, get_model_by_name

# Get token limit
limit = get_token_limit("alias-code")
print(f"Max tokens: {limit}")  # Output: Max tokens: 131072

# Get full model info
model = get_model_by_name("alias-code")
if model:
    print(f"Model: {model.name}")
    print(f"Context: {model.max_context_tokens} tokens")
```

---

## OpenCode Usage Guide

### Running OpenCode

```bash
# Start interactive session
opencode

# Run with a single message
opencode "Write a Python function to calculate fibonacci numbers"

# Use from any directory (uses global config)
opencode "Analyze the project structure"
```

### OpenCode Configuration

The configuration is at `~/.config/opencode/opencode.json`:

```json
{
  "model": "blablador/alias-code",
  "small_model": "blablador/alias-fast",
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 10000
  },
  "provider": {
    "blablador": {
      "options": {
        "baseURL": "https://api.blablador.fz-juelich.de/v1",
        "apiKey": "YOUR_TOKEN"
      },
      "models": {
        "alias-code": { "context": 98304, "output": 32768 },
        "alias-fast": { "context": 98304, "output": 32768 },
        "alias-large": { "context": 98304, "output": 32768 },
        "alias-huge": { "context": 98304, "output": 32768 }
      }
    }
  },
  "mcp": {
    "obsidian-brain": { ... },
    "markitdown": { ... },
    "docling": { ... }
  }
}
```

### Using Different Models

OpenCode will automatically use:
- **Primary model** (`alias-code`) for complex tasks
- **Small model** (`alias-fast`) for simple tasks and compaction

To use a different model explicitly:
```bash
# In OpenCode session, specify model
/model blablador:alias-huge
```

### Context Management

OpenCode automatically manages context with **Dynamic Context Pruning (DCP)**:

```json
"compaction": {
  "auto": true,          # Auto-compact context
  "prune": true,         # Remove old messages
  "reserved": 10000      # Keep 10K tokens reserved
}
```

This ensures efficient token usage and prevents context overflow.

### Obsidian Integration

OpenCode connects to your **coding-brain vault** for:

- **Knowledge management**: Store and retrieve notes
- **Context preservation**: Remember project context
- **Wikilinks**: Create connections between concepts

**Obsidian paths in config:**
- Vault location: `~/obsidian_vaults/coding-brain`
- Config file: `~/.config/opencode/opencode.json`

---

## Blablador Integration

### Blablador Models Overview

| Model Name | Alias | Context | Output | Use Case |
|------------|-------|---------|--------|----------|
| GPT-OSS-120b | `alias-code` | 98K | 32K | Code generation |
| Ministral-3-14B | `alias-fast` | 98K | 32K | Fast tasks |
| Qwen3 235 | `alias-large` | 98K | 32K | Large context |
| Meta-Llama-3.1 | `alias-huge` | 98K | 32K | Max capability |

### Token Limits

**Important:** Blablador models have:
- **Max Context**: 98,304 tokens
- **Max Output**: 32,768 tokens
- **Total Turn**: 131,072 tokens (input + output combined)

**Best Practice**: Keep conversations under 100K tokens to stay safe.

### API Endpoint

**Base URL**: `https://api.blablador.fz-juelich.de/v1`

**Authentication**: Bearer token (your API key)

### Provider Configuration

**HeLLMholtz format**: `blablador:alias-name`
**OpenCode format**: `blablador/alias-name`

Both work with the same API key and endpoint.

---

## Agent System

OpenCode comes with specialized agents for different tasks:

### Paper Assistant Agents

Located in `~/.config/opencode/agents/`:

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| `paper-assistant` | Primary coordinator | Starting any paper task |
| `paper-architect` | Structure & organization | Outlining, planning flow |
| `academic-writer` | Writing quality | Drafting, revising text |
| `latex-wizard` | LaTeX implementation | Writing .tex files |
| `citation-manager` | Bibliography | Managing .bib files |
| `figure-creator` | Visual elements | Creating plots, diagrams |
| `literature-reviewer` | Literature search | Finding and reviewing papers |

### Using Agents

```bash
# Delegating to specialized agent (in OpenCode session)
@paper-architect create an outline for a paper on quantum computing
@academic-writer draft the introduction section
@latex-wizard format this section in LaTeX
@citation-manager check my .bib file for duplicates
@figure-creator create a plot of the experimental results
@literature-reviewer find recent papers on quantum computing
```

### Agent File Structure

Each agent file (e.g., `paper-assistant.md`) contains:
- **YAML frontmatter**: Agent name, description, mode, temperature, permissions
- **Purpose**: What the agent does
- **Available tools**: MCP tools and capabilities
- **Usage guidelines**: When and how to use
- **Integration patterns**: How to work with other agents

**Example frontmatter:**
```yaml
---
name: paper-assistant
description: Scientific paper writing assistant
mode: primary
temperature: 0.3
permissions:
  read: allow
  edit: ask
  bash: ask
  websearch: allow
  task: "*": allow
---
```

### Agent Delegation Patterns

**Sequential delegation:**
```
paper-assistant → paper-architect → academic-writer → latex-wizard
```

**Parallel delegation:**
```
paper-assistant → (literature-reviewer, figure-creator, citation-manager)
```

---

### Configuration Management

#### Multiple Environments
```bash
# Project-specific config
cd ~/projects/my-project
cp .env.example .env
# Edit with project-specific settings

# User-wide default config
hellm-setup --api-key "DEFAULT_KEY"
```

**Precedence**: Project `.env` > User config > System environment

#### Model Switching
```bash
# Use different models for different tasks
hellm chat --model blablador:alias-fast "Quick question"
hellm chat --model blablador:alias-code "Write a complex function"
hellm chat --model blablador:alias-huge "Analyze this large document"
```

---

## Troubleshooting

### Common Issues

#### "Blablador API key not set"
```bash
# Solution: Configure hellm-setup
hellm-setup --api-key "YOUR_TOKEN" --base-url "https://api.blablador.fz-juelich.de/v1"

# Or drietly use .env
BLABLADOR_API_KEY=your_api_key
BLABLADOR_API_BASE=https://api.blablador.fz-juelich.de/v1
BLABLADOR_API_TIMEOUT=90
BLABLADOR_MODEL=alias-fast

HELMHOLTZ_DEFAULT_MODEL="blablador:alias-fast"

```

#### "Model not found"
```bash
# Solution: Check available models
hellm models

# Use correct prefix: blablador:model-name
hellm chat --model blablador:alias-code "Hello"
```

#### "Token limit exceeded"
```bash
# Solution: Use shorter prompts or split into multiple messages
# For OpenCode: Enable DCP (already enabled in config)
```

#### "Connection timeout"
```bash
# Solution: Check internet connection and API endpoint
curl https://api.blablador.fz-juelich.de/v1/models
```

#### "OpenCode not starting"
```bash
# Solution: Check configuration
cat ~/.config/opencode/opencode.json
# Verify API key is valid and models exist
```

### Debug Mode

```bash
# HeLLMholtz debug mode
hellm chat --model blablador:alias-code "test" --debug

# Check environment variables
echo $BLABLADOR_API_KEY
echo $BLABLADOR_API_BASE
```

### Verification Commands

```bash
# Test Blablador connection
hellm models

# Test OpenCode connection
opencode "Say hello"

# Check configuration
hellm-setup --show

# Verify API endpoint
curl -H "Authorization: Bearer $BLABLADOR_API_KEY" \
     https://api.blablador.fz-juelich.de/v1/models
```

---

## Quick Reference Cards

### HeLLMholtz CLI Quick Reference

| Task | Command |
|------|---------|
| Chat with model | `hellm chat --model blablador:alias-code "prompt"` |
| List models | `hellm models` |
| Monitor models | `hellm monitor --test-accessibility` |
| Run benchmark | `hellm bench --models blablador:alias-code --prompts-category reasoning` |
| Generate report | `hellm report --input-file results.json --output-file report.md` |
| Analyze results | `hellm analyze results.json --html-report analysis.html` |
| Throughput test | `hellm bench-throughput --model blablador:alias-code` |
| LM eval | `hellm lm-eval --model blablador:alias-code --tasks mmlu` |
| Start proxy | `hellm proxy --config config.yaml --port 8000` |

### OpenCode Quick Reference

| Task | Command |
|------|---------|
| Start session | `opencode` |
| Single query | `opencode "prompt"` |
| Switch model | `/model blablador:alias-fast` |
| Use agent | `@paper-architect create outline` |
| Check context | `/context` |
| Clear context | `/clear` |
| Save to Obsidian | `/save [topic]` |

### Blablador Model Quick Reference

| Model | Use Case | Command |
|-------|----------|---------|
| `alias-code` | Programming | `blablador:alias-code` |
| `alias-fast` | Quick tasks | `blablador:alias-fast` |
| `alias-large` | Long context | `blablador:alias-large` |
| `alias-huge` | Complex reasoning | `blablador:alias-huge` |

### Agent Quick Reference

| Task | Agent |
|------|-------|
| Paper structure | `@paper-architect` |
| Writing quality | `@academic-writer` |
| LaTeX formatting | `@latex-wizard` |
| Citations | `@citation-manager` |
| Figures | `@figure-creator` |
| Literature search | `@literature-reviewer` |
| General paper work | `@paper-assistant` |

---

## Resources

### Documentation

- **HeLLMholtz Docs**: `/home/jhe24/AID-PAIS/helmholtz_llm_suite/docs/`
- **Blablador API**: https://blablador.fz-juelich.de
- **OpenCode**: https://opencode.ai

### Configuration Files

- **HeLLMholtz Config**: `~/.config/hellmholtz/.env`
- **OpenCode Config**: `~/.config/opencode/opencode.json`
- **Agent Files**: `~/.config/opencode/agents/`
- **Obsidian Vault**: `/home/jhe24/Documents/coding-brain/`

### Important Paths

- **Project Root**: `/home/jhe24/AID-PAIS/helmholtz_llm_suite`
- **Source Code**: `/home/jhe24/AID-PAIS/helmholtz_llm_suite/src/hellmholtz/`
- **CLI Command**: `hellm` (installed in PATH)
- **Model Config**: `/home/jhe24/AID-PAIS/helmholtz_llm_suite/src/hellmholtz/providers/blablador_config.py`

---

## Getting Help

### Internal Resources

1. **Check documentation**: `/home/jhe24/AID-PAIS/helmholtz_llm_suite/docs/`
2. **Review agent files**: `~/.config/opencode/agents/`
3. **Test commands**: Use CLI examples in this guide
4. **Check config**: Verify `~/.config/opencode/opencode.json` and `.env` files

### For Questions

- **HeLLMholtz issues**: https://github.com/JonasHeinickeBio/HeLLMholtz/issues
- **OpenCode docs**: https://opencode.ai/docs
- **Blablador support**: https://blablador.fz-juelich.de/support

---

## Summary

This guide covers:

✅ **HeLLMholtz**: Unified LLM access, benchmarking, and evaluation
✅ **OpenCode**: Private AI coding assistant with Blablador
✅ **Blablador**: Helmholtz's LLM infrastructure
✅ **Agent System**: Specialized AI agents for paper writing
✅ **Best Practices**: How to use everything effectively

**Key Commands to Remember:**
```bash
# Configure once
hellm-setup --api-key "TOKEN" --base-url "https://api.blablador.fz-juelich.de/v1"

# Chat
hellm chat --model blablador:alias-code "prompt"
opencode "prompt"

# Benchmark
hellm bench --models blablador:alias-code --prompts-category reasoning

# Monitor
hellm monitor --test-accessibility
```

**Happy coding with Blablador! 🚀**
