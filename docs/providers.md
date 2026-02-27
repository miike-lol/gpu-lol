# Cloud Provider Setup

gpu-lol uses SkyPilot to abstract across providers. You never call provider APIs directly.

## RunPod (primary)

Best for: RTX3090/4090, A40, A100, H100. Consumer-grade to datacenter.

```bash
pip install runpod
runpod config  # paste API key
```

Or manually: `~/.runpod/config.toml`
```toml
[default]
api_key = "rpa_YOUR_KEY_HERE"
```

**Critical:** Must be `[default]` section — not `[credentials]`. SkyPilot silently fails otherwise.

Verify: `sky check` — RunPod should show as enabled.

RunPod API keys: https://www.runpod.io/console/user/settings

## Vast.ai (secondary — often cheaper for consumer GPUs)

Now configured via `gpu-lol secrets init` (prompts for API key → writes `~/.vast_api_key`).

Manual alternative:
```bash
vast set api-key YOUR_KEY
```

Vast API keys: https://cloud.vast.ai/account/

## Lambda Labs (tertiary — good A100 availability)

Now configured via `gpu-lol secrets init` (prompts for API key → writes `~/.lambda_cloud/lambda_keys.yaml`).

Manual alternative — create `~/.lambda_cloud/lambda_keys`:
```
[default]
api_key = YOUR_LAMBDA_KEY
```

Lambda API keys: https://cloud.lambdalabs.com/api-keys

## RunPod Templates (fast boot)

gpu-lol fetches your RunPod pod templates and auto-selects pre-cached images.
Pre-cached images boot in seconds; fresh pulls take 2-5 minutes.

```bash
gpu-lol templates              # list all templates
gpu-lol templates --filter X   # filter by name
gpu-lol up --template NAME     # pin a specific template
```

Set your preferred template family in config:
```bash
gpu-lol secrets set GPU_LOL_TEMPLATE_PATTERN=your_template_prefix
```

## Check all providers

```bash
gpu-lol check
# or
sky check
```

Shows which providers are enabled and any credential issues.

## How SkyPilot selects a provider

The generated task YAML includes:
```yaml
resources:
  accelerators: RTX4090:1
  any_of:
    - cloud: runpod
    - cloud: vast
    - cloud: lambda
```

SkyPilot tries each in order, picks first with availability at the requested spec.
If all fail (e.g., RTX3090 sold out everywhere), create a `.gpu-lol.yaml` override with a different GPU.

## GPU availability by provider

| GPU | RunPod | Vast.ai | Lambda |
|-----|--------|---------|--------|
| RTX3090 | Yes (often sold out) | Yes | Rare |
| RTX4090 | Yes | Yes | No |
| A40 | Yes | Yes | Yes |
| A100 | Yes | Yes | Yes |
| H100 | Yes | Limited | Yes |

## Pricing notes

Prices in `GPU_CATALOG` are estimates. Actual prices vary by:
- Provider and region
- Spot vs on-demand
- Current availability

SkyPilot always picks cheapest available matching the spec.
