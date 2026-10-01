# Team gateway with one-api

Run the Blablador API behind [one-api](https://github.com/songquanpeng/one-api) so a team
shares one upstream token while each person gets their own gateway token, quotas, and usage
stats.

```
team laptop ──sk-team-…──▶ one-api :3000 ──BLABLADOR_API_KEY──▶ api.helmholtz-blablador.fz-juelich.de/v1
```

If you only need a single shared endpoint (no per-user tokens or quotas), skip one-api and
run the built-in LiteLLM proxy as a service instead — see
[Connect Your Client](../../docs/connect-your-client.md#team-gateway-shared-keys).

## 1. Start the stack

```bash
cp docker-compose.yml docker-compose.yml.local   # optional: edit passwords
docker compose up -d
```

Open `http://localhost:3000` and set the root password on first visit. Change the
placeholder passwords in `docker-compose.yml` before exposing this beyond localhost.

## 2. Add the Blablador channel

In the one-api admin UI: **Channels → Add Channel**

| Field | Value |
|---|---|
| Type | **OpenAI** (Blablador is OpenAI-compatible) |
| Name | `blablador` |
| Group | `default` |
| Model | Bare ids, comma-separated, e.g. `alias-fast,alias-large,alias-code` |
| API Key | Your `BLABLADOR_API_KEY` |
| Proxy | `https://api.helmholtz-blablador.fz-juelich.de/v1` |

Two details that are easy to get wrong:

- The proxy URL must include `/v1`.
- Model names are **bare** (`alias-fast`), without any `blablador:` prefix.

Use **Test** on the channel row to verify before issuing tokens. To see the full model list:
`curl -s https://api.helmholtz-blablador.fz-juelich.de/v1/models -H "Authorization: Bearer $BLABLADOR_API_KEY"`.

## 3. Issue per-user tokens

**Tokens → Add Token** for each teammate — set a quota and allowed models. Teammates then
configure their client against the gateway:

```
Base URL:  http://<gateway-host>:3000/v1
API key:   the one-api token (sk-…)
Model:     alias-fast
```

Works for any OpenAI-compatible client (LangChain, Continue, Jan, OpenAI SDKs, …).

## 4. Claude Code users

one-api speaks the OpenAI format; Claude Code speaks Anthropic's. Per-machine Claude Code
users should use the local translation layer instead:

```bash
hellm proxy blablador:alias-fast --print-claude-settings
```

([Claude Code guide](../../docs/claude-code.md) for the full walkthrough.)

## Troubleshooting

- Channel test fails with 401 → upstream key wrong or channel proxy URL missing `/v1`.
- 404 on chat → model name carries a `provider:` prefix; strip it.
- Diagnose your HeLLMholtz install anytime with `hellm doctor`.
