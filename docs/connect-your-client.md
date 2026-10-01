# Connect Your Client

Every Blablador model speaks the OpenAI wire format, so almost any LLM client can be pointed at it. This page is the decision guide: find your client type below and follow the shortest path.

If anything fails, run the doctor first:

```bash
hellm doctor              # config, credentials, endpoint, chat round-trip
hellm doctor --skip-chat  # offline-ish checks only
```

## Which route do I need?

| Your client | Route | Command / config |
|---|---|---|
| Anything with a custom OpenAI endpoint (LangChain, curl, SDKs, web UIs) | **Direct** | Point at the Blablador API, bare model ids |
| Claude Code | **LiteLLM proxy (Anthropic format)** | `hellm proxy --print-claude-settings` |
| Claude Desktop, Cherry Studio, other MCP clients | **MCP server** | `hellm mcp --print-config` |
| Cursor / Continue / Jan / Aider / OpenCode | **Proxy or direct** | Recipes in [Blablador Integrations](blablador-integrations.md) |
| Team / shared deployment with your own keys | **Proxy in Docker** | [LiteLLM Proxy](#team-gateway-shared-keys) below |

## Route 1: Direct OpenAI-compatible endpoint

Any client that accepts a custom base URL works without HeLLMholtz running:

```
Base URL:  https://api.helmholtz-blablador.fz-juelich.de/v1
API key:   your Blablador token (BLABLADOR_API_KEY)
Model:     bare id, e.g. alias-fast — NOT blablador:alias-fast
```

Two gotchas the doctor checks for you:

- Model ids on the raw API are **bare** (`alias-fast`). The `blablador:` prefix is HeLLMholtz/aisuite syntax only; sending it upstream returns 404.
- The base URL must include `/v1`.

## Route 2: Claude Code (Anthropic format)

Claude Code speaks Anthropic's API, so run the bundled LiteLLM proxy as the translation layer:

```bash
hellm proxy blablador:alias-fast --claude-code      # prints the snippet, starts the proxy
hellm proxy blablador:alias-fast --print-claude-settings  # settings.json only, proxy not started
```

The `--print-claude-settings` output is a ready-to-paste `~/.claude/settings.json` block that hands the master key via `apiKeyHelper`, so no secret is stored in the env section. Full walkthrough: [Claude Code](claude-code.md).

## Route 3: MCP clients (Claude Desktop, ...)

HeLLMholtz ships an MCP server (tools `ask_external`, `chat_external`, `check_model`,
`run_doctor`, `list_models`, `get_info`, plus `hellm://info`/`hellm://models` resources and
`summarize`/`translate`/`explain` prompts):

```bash
hellm mcp --print-config                    # ready-to-paste Claude Desktop config
hellm mcp --transport streamable-http --port 8765   # for remote MCP clients
```

Config details and per-client placement: [Blablador Integrations §10](blablador-integrations.md#10-mcp-clients-claude-desktop-claude-code-cherry-studio).

## Route 4: Tool-specific recipes

OpenCode, Hermes, Continue.dev, Jan.AI, LangChain, GPT4All, Pi Agent, Aider, and Cursor each have a worked recipe in the [Blablador Integrations Guide](blablador-integrations.md).

## Team gateway (shared keys)

For a team, run the proxy centrally instead of distributing personal tokens:

```bash
hellm proxy blablador:alias-fast,blablador:alias-large \
    --host 0.0.0.0 --port 4000 --master-key sk-team-...
```

Client machines then use `http://<gateway>:4000/v1` (OpenAI), the Anthropic endpoint for Claude Code, or the Docker image to run the proxy as a service — see [Claude Code §Persistent configuration](claude-code.md#persistent-configuration) and `examples/one-api/` for a one-api docker-compose gateway.

## Troubleshooting checklist

1. `hellm doctor` — pinpoints the broken layer (python version, extras, key, endpoint, model id, chat).
2. `404` on chat → model id still has a `provider:` prefix, or wrong base URL.
3. `401` → key not loaded; remember `.env` in the project directory overrides shell exports.
4. Client works in terminal but not in GUI apps → the GUI never sees your shell env; put the key where the client reads it (config file / settings.json).
