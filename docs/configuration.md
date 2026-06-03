# Configuration

HeLLMholtz supports both project-level and user-level configuration.

## Project-Level Configuration (`.env`)

Create a `.env` file in your project directory:

```bash
cp .env.example .env
```

Configure your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google
GOOGLE_API_KEY=your_google_key

# Helmholtz Blablador
BLABLADOR_API_KEY=your_blablador_key
BLABLADOR_API_BASE=https://your-blablador-instance.com

# Optional: Default models
AISUITE_DEFAULT_MODELS='{"openai": "gpt-4o", "anthropic": "claude-3-haiku"}'
```

## User-Level Configuration (Recommended)

For persistent configuration across all projects (especially useful for `pipx` installations), use the `hellm-setup` tool:

### Quick Setup

```bash
# Configure once (works globally for all projects)
hellm-setup --api-key "your-blablador-key" \
            --base-url "https://api.blablador.ai/v1" \
            --default-models "meta-llama/Meta-Llama-3.1-8B-Instruct,gpt-4"
```

### Manual Setup

The configuration is stored in: `~/.config/hellmholtz/.env`

Create the directory and file:

```bash
mkdir -p ~/.config/hellmholtz
cat > ~/.config/hellmholtz/.env << EOF
BLABLADOR_API_KEY="your-blablador-api-key"
BLABLADOR_API_BASE="https://api.blablador.ai/v1"
AISUITE_DEFAULT_MODELS="meta-llama/Meta-Llama-3.1-8B-Instruct,gpt-4"
HELMHOLTZ_TIMEOUT_SECONDS=30.0
EOF
```

### CLI Configuration Options

| Option | Description |
|--------|-------------|
| `--api-key <key>` | Set Blablador API key |
| `--base-url <url>` | Set Blablador Base URL |
| `--default-models <list>` | Set default models (comma-separated) |
| `--show` | Show current configuration |

### Configuration Precedence

Configuration is loaded in this order (later overrides earlier):

1. **System environment variables** (lowest priority)
2. **User config** (`~/.config/hellmholtz/.env`)
3. **Project-local `.env`** (highest priority)

This allows you to set global defaults and override per-project.

## Configuration Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BLABLADOR_API_KEY` | Your Blablador API key | Yes (for Blablador models) |
| `BLABLADOR_API_BASE` | Base URL for the API | Yes (for Blablador models) |
| `AISUITE_DEFAULT_MODELS` | Comma-separated list of default models | No |
| `HELMHOLTZ_TIMEOUT_SECONDS` | API request timeout in seconds | No (default: 30.0) |
| `OPENAI_API_KEY` | OpenAI API key | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | No |
| `GOOGLE_API_KEY` | Google API key | No |

## Verifying Configuration

After setup, verify your configuration:

```bash
# Show current configuration
hellm-setup --show

# Test the CLI
hellm models
```

## Troubleshooting

### "Blablador API key and Base URL must be set"

1. Verify the config file exists: `ls -la ~/.config/hellmholtz/.env`
2. Check file contents: `cat ~/.config/hellmholtz/.env`
3. Ensure API key and base URL are set (not commented out)

### Environment variables not loading

Make sure your shell is loading the environment. Add to your `~/.zshrc`:

```bash
# Load hellmholtz user configuration
if [[ -f "$HOME/.config/hellmholtz/.env" ]]; then
    export $(grep -v '^#' "$HOME/.config/hellmholtz/.env" | xargs)
fi
```

## Example: Complete Setup

```bash
# 1. Navigate to your project
cd ~/projects/my-project

# 2. Run setup (one-time)
hellm-setup --api-key "sk-blablador-xyz" \
            --base-url "https://api.blablador.ai/v1" \
            --default-models "meta-llama/Meta-Llama-3.1-8B-Instruct,gpt-4"

# 3. Verify configuration
hellm-setup --show

# 4. Test models list (should work without needing .env in project)
hellm models
```
