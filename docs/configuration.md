# Configuration

HeLLMholtz is configured via environment variables. You can set these in your shell or in a `.env` file in the project root.

## Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `AISUITE_DEFAULT_MODELS` | Comma-separated list of default models to use if none specified. | `[]` |
| `HELMHOLTZ_TIMEOUT_SECONDS` | Timeout for API requests in seconds. | `30.0` |

## Blablador Settings

To use the Helmholtz Blablador API:

| Variable | Description |
|----------|-------------|
| `BLABLADOR_API_KEY` | Your Blablador API key. |
| `BLABLADOR_API_BASE` | The base URL for the Blablador API (e.g., `https://api.blablador.org/v1`). |

## Provider API Keys

Standard `aisuite` environment variables apply for other providers:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- ... and others supported by `aisuite`.

## Example `.env`

```bash
BLABLADOR_API_KEY=your_key_here
BLABLADOR_API_BASE=https://api.blablador.helmholtz.ai/v1
OPENAI_API_KEY=sk-...
HELMHOLTZ_TIMEOUT_SECONDS=60
AISUITE_DEFAULT_MODELS=openai:gpt-4o,ollama:llama3.2
```
