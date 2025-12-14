# Available Models

## Blablador Models

The following models are available via the Helmholtz Blablador API. You can list them dynamically using `hellm models`.

| ID | Model Name | Description |
|----|------------|-------------|
| `0` | **Ministral-3-14B-Instruct-2512** | The latest Ministral from Dec.2.2025 |
| `1` | **GPT-OSS-120b** | An open model released by OpenAI in August 2025 |
| `1` | **MiniMax-M2** | Our best model as of December 2025 |
| `15` | **Apertus-8B-Instruct-2509** | A new swiss model from September 2025 |
| `2` | **Qwen3 235** | A great model from Alibaba with a long context size |
| `7` | **Qwen3-Coder-30B-A3B-Instruct** | A code model from August 2025 |
| | **Devstral-Small-2-24B-Instruct-2512** | |
| | **Phi-4-multimodal-instruct** | |
| | **Qwen3-Next** | |
| | **Qwen3-VL-32B-Instruct-FP8** | |
| | **Tongyi-DeepResearch-30B-A3B** | |
| | **alias-apertus** | |
| | **alias-code** | |
| | **alias-embeddings** | |
| | **alias-fast** | |
| | **alias-function-call** | |
| | **alias-huge** | |
| | **alias-large** | |
| | **gpt-3.5-turbo** | |
| | **text-davinci-003** | |
| | **text-embedding-ada-002** | |

## Other Providers

HeLLMholtz supports any provider supported by `aisuite`, including:

- **OpenAI**: `openai:gpt-4o`, `openai:gpt-3.5-turbo`, etc.
- **Anthropic**: `anthropic:claude-3-opus-20240229`, etc.
- **Google**: `google:gemini-pro`, etc.
- **Ollama**: `ollama:llama3`, `ollama:mistral`, etc. (requires local Ollama instance)
