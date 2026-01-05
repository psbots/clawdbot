# Anthropic Bedrock Support

Clawdis supports using Claude models via AWS Bedrock as an alternative to the direct Anthropic API.

## Setup

### 1. AWS Credentials

Ensure your AWS credentials are configured. The Bedrock SDK uses the standard AWS credential chain:

- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- Shared credentials file: `~/.aws/credentials`
- IAM role (when running on AWS)

### 2. Configure models.json

Add a Bedrock provider to your `~/.pi/agent/models.json` or use Clawdis config:

```json
{
  "providers": {
    "anthropic-bedrock": {
      "baseUrl": "https://bedrock-runtime.us-east-1.amazonaws.com",
      "api": "anthropic-messages",
      "apiKey": "aws-credentials",
      "models": [
        {
          "id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
          "name": "Claude 3.5 Sonnet (Bedrock)",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": {
            "input": 3.0,
            "output": 15.0,
            "cacheRead": 0.3,
            "cacheWrite": 3.75
          },
          "contextWindow": 200000,
          "maxTokens": 8192
        },
        {
          "id": "anthropic.claude-3-opus-20240229-v1:0",
          "name": "Claude 3 Opus (Bedrock)",
          "reasoning": true,
          "input": ["text", "image"],
          "cost": {
            "input": 15.0,
            "output": 75.0,
            "cacheRead": 1.5,
            "cacheWrite": 18.75
          },
          "contextWindow": 200000,
          "maxTokens": 4096
        },
        {
          "id": "anthropic.claude-3-haiku-20240307-v1:0",
          "name": "Claude 3 Haiku (Bedrock)",
          "reasoning": false,
          "input": ["text", "image"],
          "cost": {
            "input": 0.25,
            "output": 1.25,
            "cacheRead": 0.025,
            "cacheWrite": 0.3
          },
          "contextWindow": 200000,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

### 3. Set as default (optional)

To use Bedrock by default, set in your Clawdis config:

```yaml
agent:
  provider: anthropic-bedrock
  model: anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Model IDs

Bedrock uses different model IDs than the direct Anthropic API:

| Model | Direct API ID | Bedrock ID |
|-------|--------------|------------|
| Claude 3.5 Sonnet v2 | claude-3-5-sonnet-20241022 | anthropic.claude-3-5-sonnet-20241022-v2:0 |
| Claude 3 Opus | claude-3-opus-20240229 | anthropic.claude-3-opus-20240229-v1:0 |
| Claude 3 Sonnet | claude-3-sonnet-20240229 | anthropic.claude-3-sonnet-20240229-v1:0 |
| Claude 3 Haiku | claude-3-haiku-20240307 | anthropic.claude-3-haiku-20240307-v1:0 |

## Region Configuration

By default, Bedrock uses `us-east-1`. To use a different region:

1. Set `AWS_REGION` environment variable
2. Or configure in `~/.aws/config`

The `baseUrl` in models.json should match your region:
```
https://bedrock-runtime.<region>.amazonaws.com
```

## Limitations

- Bedrock does not support prompt caching (cache tokens will always be 0)
- Bedrock does not support token counting API
- Some newer Claude features may have delayed availability on Bedrock

## Troubleshooting

### "Access denied" errors
- Ensure your IAM user/role has `bedrock:InvokeModel` permissions
- Check that Claude models are enabled in your Bedrock console

### "Model not found" errors
- Verify the model ID matches the Bedrock format exactly
- Ensure the model is available in your region

### Credential issues
- Run `aws sts get-caller-identity` to verify credentials
- Check `~/.aws/credentials` for the correct profile
