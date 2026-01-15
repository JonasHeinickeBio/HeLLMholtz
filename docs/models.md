# Available Models

## Blablador Models

The following models are available via the Helmholtz Blablador API. You can list them dynamically using `hellm models`.

### Token Limits

Each model has a maximum context window size (in tokens) that determines how much text it can process at once. HeLLMholtz provides comprehensive token limit information for all supported providers.

#### Blablador Models
- **Large Models (128k+ tokens)**: Ministral, GPT-OSS, MiniMax, Qwen3 variants, Code models, Vision models
- **Medium Models (32k tokens)**: Apertus, NVIDIA, Experimental models, Fast/Small aliases
- **Small Models (16k tokens)**: Phi-4, Legacy GPT-3.5-turbo
- **Embedding Models (8k tokens)**: Alias-embeddings, text-embedding-ada-002
- **Legacy Models (4k tokens)**: text-davinci-003

#### Other Providers
- **OpenAI**: GPT-4o (128k), GPT-4 Turbo (128k), GPT-3.5-turbo (16k)
- **Anthropic**: Claude 3 family (200k), Claude 2 (100k)
- **Google**: Gemini models (1M theoretical, often 32k-128k in practice)
- **Ollama**: Varies by model (Llama 3.2: 128k, Mistral: 32k, etc.)

You can query token limits programmatically:

```python
from hellmholtz.providers.blablador_config import get_token_limit, get_all_provider_token_limits

# Get token limit for a specific model
limit = get_token_limit("Ministral-3-14B-Instruct-2512")
print(f"Token limit: {limit}")  # Output: Token limit: 131072

# Get token limit for provider:model format
limit = get_token_limit("openai:gpt-4o")
print(f"GPT-4o token limit: {limit}")  # Output: GPT-4o token limit: 128000

# Get all token limits
all_limits = get_all_provider_token_limits()
print(all_limits["openai"]["gpt-4o"])  # Output: 128000
```

### Model List

| ID | Model Name | Description | Token Limit |
|----|------------|-------------|-------------|
| `0` | **Ministral-3-14B-Instruct-2512** | The latest Ministral from Dec.2.2025 | 128k |
| `1` | **GPT-OSS-120b** | An open model released by OpenAI in August 2025 | 128k |
| `1` | **MiniMax-M2.1** | Our best model as of December 26, 2025 | 128k |
| `15` | **Apertus-8B-Instruct-2509** | A new swiss model from September 2025 | 32k |
| `2` | **Qwen3 235** | A great model from Alibaba with a long context size | 128k+ |
| `7` | **Qwen3-Coder-30B-A3B-Instruct** | A code model from August 2025 | 128k |
| | **Devstral-Small-2-24B-Instruct-2512** | Code-focused model | 128k |
| | **Phi-4-multimodal-instruct** | Multimodal with vision | 16k |
| | **Qwen3-Next** | Latest Qwen3 model | 128k+ |
| | **Qwen3-VL-32B-Instruct-FP8** | Vision-language model | 128k+ |
| | **Tongyi-DeepResearch-30B-A3B** | Deep research model | 128k+ |
| | **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** | NVIDIA's efficient 30B model | 32k |
| | **alias-apertus** | Alias for Apertus models | 32k |
| | **alias-code** | Optimized for coding | 128k |
| | **alias-embeddings** | For text embeddings | 8k |
| | **alias-fast** | Fastest available | 32k |
| | **alias-function-call** | For function calling | 128k |
| | **alias-huge** | Largest available | 128k+ |
| | **alias-large** | Most capable | 128k+ |
| | **gpt-3.5-turbo** | Legacy GPT-3.5 | 16k |
| | **text-davinci-003** | Legacy text generation | 4k |
| | **text-embedding-ada-002** | Legacy embeddings | 8k |

> **Note**: Some models share the same ID (e.g., both GPT-OSS-120b and MiniMax-M2.1 use ID `1`). This is intentional and allows multiple models to be accessed via the same identifier.

## Other Providers

HeLLMholtz supports any provider supported by `aisuite`, including:

- **OpenAI**: `openai:gpt-4o`, `openai:gpt-3.5-turbo`, etc.
- **Anthropic**: `anthropic:claude-3-opus-20240229`, etc.
- **Google**: `google:gemini-pro`, etc.
- **Ollama**: `ollama:llama3`, `ollama:mistral`, etc. (requires local Ollama instance)
