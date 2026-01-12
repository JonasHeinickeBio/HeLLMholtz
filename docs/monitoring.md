# Model Monitoring

HeLLMholtz includes comprehensive monitoring capabilities for tracking model availability and configuration consistency, particularly for Blablador models.

## Overview

The monitoring system provides:

- **Model Availability Tracking**: Check which models are currently available via the API
- **Configuration Consistency**: Ensure your local configuration matches available models
- **Accessibility Testing**: Test actual model accessibility with real API calls
- **Health Monitoring**: Monitor API endpoints and model status

## Python API

### Basic Monitoring

```python
from hellmholtz.monitoring import ModelAvailabilityMonitor

# Initialize monitor (uses environment variables)
monitor = ModelAvailabilityMonitor()

# Get currently available models from API
api_models = monitor.get_api_models()
print(f"Available models: {len(api_models)}")

# Get configured models from local config
config_models = monitor.get_configured_models()
print(f"Configured models: {len(config_models)}")
```

### Testing Model Accessibility

```python
# Test if a specific model is accessible
is_accessible, latency = monitor.test_model_accessibility("gpt-4o")
print(f"Model accessible: {is_accessible}, Latency: {latency:.2f}s")

# Analyze overall availability
analysis = monitor.analyze_availability(test_accessibility=True)
print(analysis)
```

### Custom Configuration

```python
# Use custom API credentials
monitor = ModelAvailabilityMonitor(
    api_key="your-api-key",
    api_base="https://your-blablador-instance.com/v1"
)
```

## Command Line Interface

### Basic Model Listing

```bash
# List all available models
hellm models
```

### Model Monitoring

```bash
# Monitor model availability
hellm monitor

# Test actual model accessibility (makes real API calls)
hellm monitor --test-accessibility

# Check configuration consistency
hellm monitor --check-config
```

### Output Example

```
🔍 Model Availability Analysis
══════════════════════════════════════════════

📊 API Models Found: 15
📋 Configured Models: 12

✅ Available Models:
  • gpt-4o (accessible, 0.85s latency)
  • claude-3-sonnet (accessible, 1.12s latency)
  • gemini-pro (accessible, 0.92s latency)

⚠️  Configuration Issues:
  • Model 'old-model-v1' not found in API
  • Model 'deprecated-model' marked as inactive

📈 Health Summary:
  • Overall Success Rate: 92.3%
  • Average Latency: 0.96s
  • Total Tests: 15
```

## Configuration

The monitoring system uses the same environment variables as the main client:

```bash
# Required
BLABLADOR_API_KEY=your-api-key
BLABLADOR_API_BASE=https://api.blablador.example.com/v1

# Optional
HELMHOLTZ_TIMEOUT_SECONDS=30
```

## Integration with CI/CD

You can integrate monitoring into your CI/CD pipelines:

```bash
#!/bin/bash
# Health check script
hellm monitor --test-accessibility --json > health_report.json

# Check if critical models are available
if ! jq '.summary.overall_success_rate > 0.95' health_report.json; then
    echo "Model health check failed!"
    exit 1
fi
```

## Troubleshooting

### Common Issues

**Connection Timeout**
```bash
# Increase timeout
export HELMHOLTZ_TIMEOUT_SECONDS=60
hellm monitor --test-accessibility
```

**API Key Issues**
```bash
# Verify API key is set
echo $BLABLADOR_API_KEY
# Should show your key (keep it secret!)
```

**Model Not Found**
- Check if the model name/alias is correct
- Verify the model is still available in the API
- Update your configuration if needed

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

monitor = ModelAvailabilityMonitor()
# Now you'll see detailed API requests/responses
```
