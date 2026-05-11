# android-studio-llm-openai-reasoning-proxy

**A Windows-first local intelligence service that connects Android Studio agents to OpenAI-compatible reasoning models while preserving reasoning state, enriching project context, routing tasks intelligently, and protecting developer workflows.**

---

## The Problem

Android Studio's AI agent (Gemini, GitHub Copilot, or any OpenAI-compatible plugin) does not preserve provider-specific reasoning metadata between turns. When using models that return `reasoning_content` (DeepSeek R1, OpenAI `o` series), the next request is sent without that field, causing errors like:

```
Error: com.openai.errors.BadRequestException: 400:
The reasoning_content in the thinking mode must be passed back to the API.
```

Similar errors occur with Anthropic thinking blocks, and any provider whose models require state passed back from prior responses.

---

## What This Service Does

```
Android Studio -> [this proxy :3001] -> LLM Provider API
```

1. **Intercepts** every chat completion request from Android Studio.
2. **Detects** whether the target model requires reasoning state.
3. **Repairs** missing `reasoning_content` / `thinking` blocks from the local encrypted cache.
4. **Forwards** the corrected request to the upstream provider.
5. **Captures** reasoning state from the response and stores it in the cache.
6. **Streams** the response back to Android Studio transparently.

---

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for your LLM provider

### Install

```powershell
git clone https://github.com/yourname/android-studio-llm-openai-reasoning-proxy
cd android-studio-llm-openai-reasoning-proxy
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run (two options)

**Option A -- Desktop shortcut (recommended)**

Run the shortcut setup script once:

```powershell
powershell -ExecutionPolicy Bypass -File setup_shortcut.ps1
```

This places an **ASLOR Proxy** shortcut on your desktop.  Double-click it to start
the server.  A console window opens showing the live log output; press Ctrl+C to
stop.

**Option B -- Command line**

```powershell
python -m aslor.main
```

The proxy is now available at `http://127.0.0.1:3001`.

### Auto Reload

The proxy can now restart itself automatically when you edit:

- Python files under the project
- `config.yaml`
- `models.yaml`
- `skills.yaml`
- `.env` and `.env.local`

The Windows launcher `start.bat` enables this by default.  From the command
line, you can also turn it on explicitly:

```powershell
python -m aslor.main --reload
```

Or via environment variable:

```powershell
$env:ASLOR_RELOAD="1"
python -m aslor.main
```

If you keep provider secrets in a `.env` file, the app now loads them on
startup and after each reload.

### Point Android Studio at the Proxy

In your Android Studio AI plugin settings, set the **API Base URL** to:

```
http://127.0.0.1:3001/v1
```

The proxy forwards the API key you configure in Android Studio directly to the
upstream provider.  Set your real API key in Android Studio's settings and the
proxy will pass it through.  If you omit the key in Android Studio, the proxy
falls back to the `api_key_env` environment variable from `config.yaml`.

---

## Configuration Dashboard

Once the server is running, open your browser to:

```
http://127.0.0.1:3001/dashboard
```

A built-in configuration panel lets you change every setting without editing
YAML by hand:

| Section | Fields |
|---------|--------|
| **Provider** | Provider name, base URL, API key env var name, timeout |
| **Server** | Bind host, port |
| **Cache & Logging** | Cache path, encryption toggle, log level, log format |

Click **Save Configuration** to write changes to `config.yaml`, then restart the
server for them to take effect.

---

## Configuration File (`config.yaml`)

The dashboard writes this file for you, but you can also edit it directly.

```yaml
server:
  host: 127.0.0.1
  port: 3001

provider:
  name: deepseek         # openai | deepseek | anthropic | passthrough
  base_url: https://api.deepseek.com/v1
  api_key_env: DEEPSEEK_API_KEY
  timeout_seconds: 120

cache:
  path: ./data/cache.db
  encrypt: true

logging:
  level: INFO            # DEBUG | INFO | WARNING | ERROR
  format: json           # json | text

vision:
  enabled: false
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  timeout_seconds: 60
  upload_dir: ./data/vision
  max_image_bytes: 10485760
```

### Vision Sidecar

If your main chat model is text-only, the proxy can still analyze screenshots
through a separate vision-capable model.

1. Enable the `vision` block in `config.yaml`.
2. Upload a screenshot to `POST /admin/vision/images` using JSON with
   `filename`, `mime_type`, and `content_base64`.
3. Use the returned `aslor://image/<id>` reference inside a chat message.
4. The proxy calls the vision sidecar, converts the result into
   `VISUAL_CONTEXT`, and injects that into the normal request before forwarding.

This lets Android UI screenshots participate in debugging even when the main
reasoning model cannot accept images directly.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Primary proxy endpoint -- repairs reasoning state, forwards, streams back |
| `GET` | `/v1/models` | Returns model list from upstream |
| `GET` | `/status` | Service health, cache stats, uptime |
| `GET` | `/dashboard` | Built-in configuration UI |
| `GET` | `/admin/config` | Read current config as JSON |
| `PUT` | `/admin/config` | Write config to `config.yaml` |
| `POST` | `/admin/cache/clear` | Clear reasoning state cache |
| `POST` | `/admin/vision/images` | Upload a screenshot and get an `aslor://image/...` reference |

---

## Provider Support

| Provider | Adapter | Reasoning Field | Streaming |
|----------|---------|-----------------|-----------|
| OpenAI (`o1`, `o3`, `o4`) | `openai` | `reasoning_content` in assistant message | Yes |
| DeepSeek (`deepseek-reasoner`) | `deepseek` | `reasoning_content` in assistant message | Yes |
| Anthropic (`claude-3-7-sonnet`) | `anthropic` | `thinking` content block | Yes |
| Chutes (`Qwen/...`, `GLM-5-TEE`, `Qwen2.5-VL`, etc.) | `chutes` | configurable via model registry | Yes |
| Any OpenAI-compatible | `passthrough` | Safe passthrough, no state repair | Yes |

---

## Reasoning State Repair -- How It Works

### The Root Cause

OpenAI `o` series and DeepSeek reasoner models return an assistant message with two parts:

```json
{
  "role": "assistant",
  "content": "Here is my answer...",
  "reasoning_content": "<internal chain of thought>"
}
```

When Android Studio sends the next turn, it serializes the conversation history but **strips** `reasoning_content` because it is not in the base OpenAI spec it knows about.

The upstream model then rejects the request because it sees the assistant message is missing the required field.

### The Fix

This proxy caches the full assistant message (with `reasoning_content`) keyed by a session ID derived from the conversation. On the next request, it inspects assistant messages and re-injects any missing reasoning fields before forwarding.

```
Request in -> detect model -> look up cached state -> inject missing fields -> forward -> capture response state -> cache -> stream back
```

---

## Security

- API keys are **never** stored in the cache or logs.
- Client-supplied `Authorization` headers are passed through to the upstream
  provider and never persisted.
- The cache database is **Fernet-encrypted** at rest.
- Request and response logs redact any string matching common secret patterns.
- The service binds to `127.0.0.1` by default (localhost only).

---

## Architecture

```
aslor/
  main.py               Entry point, lifespan, ASGI app
  config.py             YAML config loader + env expansion
  server/
    app.py              FastAPI app, route registration
    routes.py           /v1/chat/completions, /status, /dashboard, config CRUD
    dashboard.html      Built-in configuration UI (zero dependencies)
    middleware.py       Request ID, logging, secret redaction
  providers/
    base.py             Abstract adapter interface
    openai.py           OpenAI reasoning adapter
    deepseek.py         DeepSeek reasoning adapter
    anthropic.py        Anthropic thinking adapter
    passthrough.py      Safe generic adapter
    registry.py         Adapter selection by config / model name
  reasoning/
    state.py            ReasoningStateStore (encrypted SQLite)
    repair.py           Injects missing reasoning blocks into messages
    detector.py         Detects whether a model needs state repair
  cache/
    db.py               SQLite session with optional Fernet encryption
  models/
    request.py          Pydantic models for incoming requests
    response.py         Pydantic models for outgoing responses
  logging_config.py     Structured safe logging (no secrets)
tests/
  test_repair.py        Reasoning state repair unit tests
  test_adapters.py      Provider adapter unit tests
  test_deepseek.py      DeepSeek-specific integration tests
  test_routes.py        FastAPI route integration tests
  test_cache.py         Cache encryption/decryption tests
config.example.yaml
start.bat              Double-click launcher for Windows
setup_shortcut.ps1     Desktop shortcut creator
requirements.txt
```

---

## Development

```powershell
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## License

MIT
