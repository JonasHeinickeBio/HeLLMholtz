# Blablador Integrations Guide

This guide covers how to connect [Helmholtz Blablador](https://blablador.fz-juelich.de) to various AI coding tools and frameworks. All tools are open-source and work with Blablador's private models.

## Quick Setup

Use the `hellm-setup` CLI for one-time configuration:

```bash
# Set your API key once
hellm-setup --set-api-key YOUR_TOKEN

# Export config to any supported tool
hellm manager export opencode
hellm manager export hermes
hellm manager export continue
# ... etc
```

List all supported tools:

```bash
hellm manager tools
```

---

## Supported Tools

### 1. OpenCode

**Type:** CLI AI coding assistant (like Claude Code, but open-source and private)

**Config location:** `~/.config/opencode/opencode.json`

**Setup:**
1. Install OpenCode from [opencode.ai](https://opencode.ai)
2. Run it once to generate config files, then close
3. Export config:

```bash
hellm manager export opencode
```

**Or manually add to `~/.config/opencode/opencode.json`:**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@tarquinen/opencode-dcp@latest"],
  "model": "blablador/alias-fast",
  "small_model": "blablador/alias-fast",
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 10000
  },
  "provider": {
    "blablador": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Blablador",
      "options": {
        "baseURL": "https://api.blablador.fz-juelich.de/v1",
        "apiKey": "YOUR_BLABLADOR_API_KEY_HERE"
      },
      "models": {
        "alias-code": {
          "name": "Blablador Code (alias-code)",
          "limit": { "context": 98304, "output": 32768 }
        },
        "alias-huge": {
          "name": "Blablador Minimax (alias-huge)",
          "limit": { "context": 98304, "output": 32768 }
        },
        "alias-fast": {
          "name": "Blablador Fast (alias-fast)",
          "limit": { "context": 98304, "output": 32768 }
        },
        "alias-large": {
          "name": "Blablador Large (alias-large)",
          "limit": { "context": 98304, "output": 32768 }
        }
      }
    }
  }
}
```

---

### 2. Hermes Agent

**Type:** Full-featured CLI AI agent with memory, skills, and subagent delegation

**Config location:** `~/.hermes/config.json`

**Setup:**
1. Install Hermes from [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com)
2. During setup, choose "Custom Endpoint" and enter:
   - Base URL: `https://api.blablador.fz-juelich.de/v1`
   - API Key: your Blablador token
   - Model alias: `alias-code`
3. Export config:

```bash
hellm manager export hermes
```

**Optional:** Set up web search skill by pasting in chat:
```
Add a new skill for web search. It will use the searxng search engine with the following URL: https://search.blablador.fz-juelich.de - and this will be your MAIN AND ONLY web search.
```

---

### 3. Continue.dev (VSCode / Sublime Text)

**Type:** AI code assistant extension for VSCode and Sublime Text

**Config location:** `~/.continue/config.yaml`

**Setup:**
1. Install the Continue.dev extension in VSCode
2. Click the Continue icon → Gear → Open Config file
3. Export config:

```bash
hellm manager export continue
```

**Or manually fill `config.yaml`:**

```yaml
name: Blablador Helmholtz
version: 1.0.0
schema: v1
models:
  - name: Blablador
    provider: openai
    model: AUTODETECT
    apiKey: YOUR_BLABLADOR_TOKEN
    apiBase: https://api.blablador.fz-juelich.de/v1
context:
  - provider: code
  - provider: docs
  - provider: diff
  - provider: terminal
  - provider: problems
  - provider: folder
  - provider: codebase
```

---

### 4. Jan.AI

**Type:** Private desktop AI assistant with local + cloud model support

**Config location:** Jan UI (Settings → Model Providers)

**Setup:**
1. Download [Jan.AI](https://jan.ai)
2. Open Settings → Model Providers → Add Provider
3. Enter:
   - Name: `Blablador`
   - Base URL: `https://api.blablador.fz-juelich.de/v1`
   - API Key: your Blablador token
4. Click "Refresh" to load models
5. Or export config:

```bash
hellm manager export jan
```

---

### 5. LangChain

**Type:** Python framework for RAG and document-based Q&A

**Config:** Environment variables

**Setup:**
```bash
# Create environment
uv venv langchain
source langchain/bin/activate
uv pip install langchain_openai langchain_community langchain chromadb
```

**Export env vars:**
```bash
hellm manager export langchain
source ~/.config/hellmholtz/langchain.env
```

**Or set manually:**
```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_BLABLADOR_TOKEN"
os.environ["OPENAI_API_BASE"] = "https://api.blablador.fz-juelich.de/v1"
```

**Example RAG script:**
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_classic.indexes import VectorstoreIndexCreator

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
loader = TextLoader("your_document.txt")
index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
llm = ChatOpenAI(model="alias-fast")

answer = index.query("Your question here", llm=llm)
print(answer)
```

---

### 6. GPT4All

**Type:** Private local chatbot with document support (LocalDocs)

**Config:** GUI setup (reference config available)

**Setup:**
1. Download [GPT4All](https://www.nomic.ai/gpt4all)
2. Go to Models → Add Model → OpenAI Compatible
3. Enter:
   - API Key: your Blablador token
   - Base URL: `https://api.blablador.fz-juelich.de/v1`
   - Model Name: `alias-fast`
4. Or export reference config:

```bash
hellm manager export gpt4all
cat ~/.config/hellmholtz/gpt4all-reference.json
```

**Tip:** Use the LocalDocs feature to chat with your PDF files privately.

---

### 7. Pi Agent

**Type:** Minimal terminal coding harness

**Config location:** `~/.pi/agent/models.json`

**Setup:**
1. Install Pi from [pi.dev](https://pi.dev/)
2. Edit `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "blablador": {
      "baseUrl": "https://api.helmholtz-blablador.fz-juelich.de/v1",
      "api": "openai-completions",
      "apiKey": "YOUR_BLABLADOR_API_TOKEN",
      "models": [
        { "id": "alias-code" },
        { "id": "alias-huge" },
        { "id": "alias-fast" },
        { "id": "alias-large" }
      ]
    }
  }
}
```

3. Or export config:

```bash
hellm manager export pi
```

4. Start Pi and use `/model` to select Blablador models

**Extend Pi:** Install packages like `pi-web-access` for web browsing:
```bash
pi install npm:pi-web-access
```

---

### 8. Aider

**Type:** AI pair programming in your terminal

**Config location:** `~/.aider.conf.yml`

**Setup:**
```bash
hellm manager export aider
```

**Or manually create `~/.aider.conf.yml`:**
```yaml
model: openai/alias-code
openai-api-key: YOUR_BLABLADOR_TOKEN
openai-api-base: https://api.blablador.fz-juelich.de/v1
```

---

### 9. Cursor

**Type:** AI-first code editor

**Config location:** `~/.cursor/.env`

**Setup:**
```bash
hellm manager export cursor
```

**Or manually set in `~/.cursor/.env`:**
```
OPENAI_API_KEY=YOUR_BLABLADOR_TOKEN
OPENAI_BASE_URL=https://api.blablador.fz-juelich.de/v1
```

---

## Available Blablador Models

| Alias | Description | Context | Max Output |
|-------|-------------|---------|------------|
| `alias-code` | Optimized for code generation | 98,304 | 32,768 |
| `alias-fast` | Fast responses for quick tasks | 98,304 | 32,768 |
| `alias-large` | Large context window | 98,304 | 32,768 |
| `alias-huge` | Maximum capability (Minimax) | 98,304 | 32,768 |

Models are subject to change without notice, but aliases are preserved.

## Getting Your Blablador Token

1. Visit [blablador.fz-juelich.de](https://blablador.fz-juelich.de)
2. Log in with your Helmholtz credentials
3. Generate an API token
4. Use this token as your API key in any tool above

## Need Help?

- [Blablador API Access Guide](https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/)
- [AI@JSC Documentation](https://sdlaml.pages.jsc.fz-juelich.de/)
