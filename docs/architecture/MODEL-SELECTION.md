# Model Selection Guide

## How to Choose Models for Different Providers

### Overview

Token Miser allows you to configure which model is used for each provider through the `model_mapping` configuration.

### Configuration File (`config.toml`)

Each provider has a `model_mapping` section:

```toml
[providers.github_copilot.model_mapping]
default = "gpt-4o"
"gpt-4" = "gpt-4"
"claude-3-sonnet" = "gpt-4o"
```

### How Model Selection Works

1. **User requests a model**: e.g., `{"model": "claude-3-opus", ...}`
2. **Proxy looks up mapping**: Checks `model_mapping` for the provider
3. **Falls back to default**: If model not in mapping, uses `default`
4. **Forwards to provider**: Sends request with mapped model name

### GitHub Copilot Subscription

**Available Models** (depends on your subscription tier):
- **Copilot Individual**: `gpt-4o`, `gpt-4o-mini`
- **Copilot Business**: `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- **Copilot Enterprise**: `gpt-4`, `gpt-4o` + advanced features

**Configuration Example**:

```toml
[providers.github_copilot]
endpoint = "https://api.githubcopilot.com"
auth_type = "GitHubOAuth"

[providers.github_copilot.model_mapping]
# Use GPT-4o for most requests (best quality in Copilot)
default = "gpt-4o"

# Explicit mappings
"gpt-4" = "gpt-4"
"gpt-4o" = "gpt-4o"
"gpt-4o-mini" = "gpt-4o-mini"

# Map Claude/OpenAI requests to Copilot equivalents
"claude-3-opus" = "gpt-4o"      # Best Copilot model for complex tasks
"claude-3-sonnet" = "gpt-4o"    # High quality standard model
"claude-3-haiku" = "gpt-4o-mini"  # Fast, efficient model
```

**Limitations**:
- GitHub Copilot **auto-selects** models based on context
- You can't directly control which model Copilot uses
- Model availability varies by subscription tier
- No Claude models available in Copilot (must map to GPT)

### Anthropic API (Claude)

**Available Models** (examples):
- `claude-sonnet-4` (default)
- `claude-opus-4`

**Configuration Example**:

```toml
[providers.tier3_complex]
endpoint = "https://api.anthropic.com/v1"
auth_type = "ApiKey"
api_key = "${ANTHROPIC_API_KEY}"

[providers.tier3_complex.model_mapping]
default = "claude-sonnet-4"

# Map GPT requests to Claude equivalents
"gpt-4" = "claude-sonnet-4"
"gpt-4-turbo" = "claude-opus-4"
"gpt-4o" = "claude-sonnet-4"
```

> Note: the raw Anthropic API is not OpenAI-format. Point `tier3_complex` at an
> OpenAI-compatible gateway (OpenRouter / DeepSeek / z.ai / vLLM), not directly at
> `api.anthropic.com`.

### Private/Enterprise Cluster

**Available Models** (depends on cluster configuration):
- `llama-3.1-70b` (standard inference)
- `deepseek-coder-33b` (complex reasoning)
- `llama-3.2-1b` (intent classification)

**Configuration Example**:

```toml
[providers.tier2_standard]
endpoint = "https://llm-cluster.internal.example.com/v1"
auth_type = "None"

[providers.tier2_standard.model_mapping]
default = "llama-3.1-70b"

# Use deepseek for complex coding tasks
"gpt-4" = "deepseek-coder-33b"
"claude-3-opus" = "deepseek-coder-33b"

# Use Llama for standard tasks
"claude-3-sonnet" = "llama-3.1-70b"
"gpt-4o-mini" = "llama-3.1-70b"
```

### Tier-Based Model Selection

**Problem**: Want different models for different task complexities.

**Solution**: Configure each tier separately:

```toml
# Tier 1 (Trivial) - Use fast, free models
[providers.tier1_private]
endpoint = "https://llm-cluster.internal.example.com/v1"

[providers.tier1_private.model_mapping]
default = "llama-3.1-70b"

# Tier 2 (Standard) - Use efficient mid-tier
[providers.tier2_standard]
endpoint = "https://api.openai.com/v1"
auth_type = "ApiKey"

[providers.tier2_standard.model_mapping]
default = "gpt-4o-mini"

# Tier 3 (Complex) - Use premium models
[providers.tier3_complex]
endpoint = "https://api.anthropic.com/v1"
auth_type = "ApiKey"

[providers.tier3_complex.model_mapping]
default = "claude-3-opus-20240229"
```

### Runtime Model Override

**Scenario**: User wants to override default model selection.

**Option 1: Request Header** (not yet implemented):

```bash
curl -H "X-Preferred-Model: claude-3-opus" \
  http://localhost:8080/v1/chat/completions
```

**Option 2: Request Parameter** (not yet implemented):

```json
{
  "model": "auto",
  "messages": [...],
  "provider_hints": {
    "preferred_model": "claude-3-opus"
  }
}
```

**Option 3: Multiple Endpoints** (recommended):

Run multiple proxy instances on different ports:

```bash
# Port 8080: Default routing
./target/release/token_miser --port 8080 --config config.toml

# Port 8081: Force premium models
./target/release/token_miser --port 8081 --config config-premium.toml

# Port 8082: Force local models
./target/release/token_miser --port 8082 --config config-local.toml
```

Then configure your client:

```bash
# Use premium models
ANTHROPIC_BASE_URL=http://localhost:8081 claude

# Use free models
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

### Best Practices

1. **Use `default` wisely**: Set to most commonly used model
2. **Map equivalents**: Map across providers (Claude ↔ GPT ↔ Llama)
3. **Consider costs**: Use cheaper models for simple tasks
4. **Test mappings**: Verify model availability before deploying
5. **Document choices**: Comment why you chose specific mappings

### Example: Full Configuration

```toml
# Premium cloud model for the complex tier
[providers.tier3_complex]
endpoint = "https://openrouter.ai/api/v1"
auth_type = "ApiKey"
api_key = "${OPENROUTER_API_KEY}"
priority = 1

[providers.tier3_complex.model_mapping]
default = "anthropic/claude-sonnet-4"

# But falls back to a private cluster if the cloud tier is unavailable
[providers.tier3_fallback]
endpoint = "https://llm-cluster.internal.example.com/v1"
auth_type = "None"
priority = 2

[providers.tier3_fallback.model_mapping]
default = "deepseek-coder-33b"
"claude-3-opus" = "deepseek-coder-33b"
```

### Limitations

1. **No runtime switching**: Must restart proxy to change config
2. **Static mapping**: Can't dynamically select based on request content
3. **Provider models**: Can't use models the provider doesn't support
4. **Subscription tiers**: Available models depend on your subscription

### Future Features (Not Yet Implemented)

- [ ] Dynamic model selection via API
- [ ] Per-request model override headers
- [ ] Automatic model discovery from provider
- [ ] Cost-based automatic switching
- [ ] User preference learning
