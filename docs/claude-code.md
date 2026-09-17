# Claude Code via the LiteLLM Proxy

HeLLMholtz can run a local LiteLLM proxy that speaks **both** the OpenAI API
(`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`). This lets you point
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) at any model
HeLLMholtz can route — OpenAI, Gemini, Ollama, Blablador, and more — without
an Anthropic API key.

## How it works

Claude Code is an Anthropic-API client: it sends requests to
`https://api.anthropic.com/v1/messages` and authenticates with a Bearer token.
Two environment variables change that behavior:

| Variable | Effect |
| -------- | ------ |
| `ANTHROPIC_BASE_URL` | Where Claude Code sends requests |
| `ANTHROPIC_AUTH_TOKEN` | Bearer token sent on every request |
| `ANTHROPIC_MODEL` | Default model name sent in requests |

Point them at a server that implements the Anthropic Messages API, and Claude
Code just works. LiteLLM implements `/v1/messages` by translating Anthropic
requests into the target model's native API.

> **Why can't I use a plain OpenAI endpoint directly?**
> Claude Code does **not** speak the OpenAI chat-completions API — only the
> Anthropic Messages API. A plain OpenAI-compatible endpoint (or `litellm`
> without the `/v1/messages` route) returns `404` for Claude Code's requests.
> The LiteLLM proxy is the missing translation layer.

## Quickstart

Requires the `[proxy]` extra (`pip install .[proxy]`) and the upstream model's
API key in your environment (e.g. `OPENAI_API_KEY`).

```bash
# Terminal 1: start the proxy
hellm proxy openai:gpt-4o --name claude --claude-code
```

The proxy auto-generates a master key (prefix `sk-hellm-`) and prints a ready
shell snippet:

```text
Starting LiteLLM proxy for openai:gpt-4o (alias: claude) on http://127.0.0.1:4000
  OpenAI-compatible:   http://127.0.0.1:4000/v1/chat/completions
  Anthropic-compatible: http://127.0.0.1:4000/v1/messages

Point Claude Code at the proxy (run in a second terminal):
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL="http://127.0.0.1:4000"
export ANTHROPIC_AUTH_TOKEN="sk-hellm-…"
export ANTHROPIC_MODEL="claude"
claude
```

```bash
# Terminal 2: run the printed snippet, then start Claude Code
claude
```

`--name claude` is optional — without it, the alias is the full model string
(`openai:gpt-4o`), which is what Claude Code sends as the model name.

## Manual master key

Generate or provide your own key instead of the auto-generated one:

```bash
# Explicit key
hellm proxy openai:gpt-4o --name claude --master-key sk-my-own-key

# Or via environment variable (the printed snippet references it, never the value)
export LITELLM_MASTER_KEY=sk-my-own-key
hellm proxy openai:gpt-4o --name claude --claude-code
```

Key resolution order: `--master-key` → `LITELLM_MASTER_KEY` env var →
auto-generated (only in `--claude-code` mode) → no auth (open proxy, warning
printed).

## Verifying the proxy

List the models exposed by the proxy:

```bash
curl -s http://127.0.0.1:4000/v1/models \
  -H "Authorization: Bearer sk-hellm-…" | python3 -m json.tool
```

Send a raw Anthropic-format request:

```bash
curl -s http://127.0.0.1:4000/v1/messages \
  -H "Authorization: Bearer sk-hellm-…" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude", "max_tokens": 50, "messages": [{"role": "user", "content": "Hi"}]}'
```

A `404` here means the proxy isn't running with the Anthropic route; a
provider error (e.g. upstream `401` for a bad `OPENAI_API_KEY`) means the
route exists and the problem is upstream.

## Persistent configuration

The exported environment variables only last for the current shell. For a
permanent setup, put them in Claude Code's user-level settings:

```json
// ~/.claude/settings.json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-hellm-…",
    "ANTHROPIC_MODEL": "claude"
  }
}
```

**Keep secrets out of project files.** User-level settings (`~/.claude/`)
apply to all your projects without risking a key being committed to a
repository. If your team shares a proxy, give each member their own key.

> **Gotcha:** if `ANTHROPIC_API_KEY` is set in your environment, Claude Code
> uses it **instead of** `ANTHROPIC_AUTH_TOKEN`. The printed snippet starts
> with `unset ANTHROPIC_API_KEY` for this reason — keep it if you copy the
> snippet manually.

## Choosing a model

- Use an instruction-tuned chat model; `--name` sets the alias Claude Code
  requests.
- One proxy instance serves one model. To switch, stop the proxy and start a
  new one with a different model/`--name` (e.g. `gemini:gemini-2.5-pro`,
  `ollama:llama3.2`, `blablador:…`).
- Claude Code's small/fast background model also uses `ANTHROPIC_MODEL` via
  the proxy, so the same model handles everything.

## Caveats

- **MCP and Anthropic-specific features:** the core chat + tool-calling loop
  works through the proxy. Features that depend on Anthropic-only endpoints
  (some MCP flows, extended thinking) may be limited when the proxy fronts a
  non-Anthropic model.
- **Localhost binding:** the default `--host 127.0.0.1` keeps the proxy on
  your machine. Only bind to other interfaces if the proxy is authenticated
  (master key set) and on a trusted network.
- **The master key is a secret:** with a master key set, requests without it
  are rejected. Treat the printed key like an API key — don't paste it into
  commits or shared docs.
- **Rejected requests may return `500` instead of `401/403`:** with a master
  key configured, requests that *fail* authentication (missing or wrong key)
  currently surface as a `500 Internal Server Error` rather than a clean
  `401`. Valid-key requests are unaffected and work normally. This is a bug in
  the pinned `litellm` version (its auth-error path tries to import the
  optional `prisma` dependency); it is fixed upstream. Watch the proxy log for
  the underlying `401`-style message if you need to distinguish a bad key from
  a real error.

## All proxy flags

```text
hellm proxy MODEL [OPTIONS]

Options:
  --port INTEGER        Port to listen on (default: 4000)
  --host TEXT           Host to bind the proxy to (default: 127.0.0.1)
  --name TEXT           Alias to expose the model under
  --master-key TEXT     Proxy master key (or set LITELLM_MASTER_KEY)
  --config PATH         Use an existing LiteLLM config file
  --claude-code         Print the Claude Code snippet; auto-generate a master key
  --debug               Run the proxy in debug mode
```
