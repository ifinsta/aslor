# AGENTS.md

This file describes the internal agents and intelligence layers of the
`android-studio-llm-openai-reasoning-proxy` service.

Each agent is a focused, loosely-coupled module with a single responsibility.
Agents communicate through well-defined interfaces, not direct imports across
boundaries.

---

## Agent Map

```
Android Studio request
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  RequestPipelineAgent                                       │
│  Orchestrates the full request lifecycle                    │
│                                                             │
│  1. SecretRedactionAgent    (sanitize before logging)       │
│  2. ModelDetectorAgent      (which provider + model)        │
│  3. ReasoningRepairAgent    (inject missing state)          │
│  4. ProviderRouterAgent     (select upstream adapter)       │
│  5. UpstreamForwardAgent    (send to LLM provider)          │
│  6. ResponseCaptureAgent    (extract + cache reasoning)     │
│  7. StreamRelayAgent        (stream chunks to client)       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Android Studio response
```

---

## Agent Descriptions

### RequestPipelineAgent

**File:** `aslor/pipeline.py`

The central orchestrator. Receives the raw request from the FastAPI route,
calls each sub-agent in order, and returns the final streaming or non-streaming
response.

Does not contain business logic itself. It only wires agents together.

**Inputs:** raw HTTP request body, headers  
**Outputs:** streaming or non-streaming HTTP response  
**Failure mode:** if any agent raises `PipelineError`, returns a safe 500 with no
sensitive content.

---

### SecretRedactionAgent

**File:** `aslor/agents/redaction.py`

Scans request and response bodies for strings that match known secret patterns:

- Bearer tokens
- API keys (`sk-`, `pk-`, `ghp_`, etc.)
- Firebase config secrets
- `.env` style `KEY=VALUE` patterns
- Anything matching `[A-Za-z0-9_]{20,}` adjacent to known key name prefixes

Replaces matched values with `[REDACTED]` in log output only.
**Never modifies** the actual request payload.

**Inputs:** serialized request/response dict  
**Outputs:** sanitized dict (for logging only)  
**Side effects:** none on the real request

---

### ModelDetectorAgent

**File:** `aslor/agents/detector.py`

Determines:

1. Which provider adapter to use (OpenAI, DeepSeek, Anthropic, passthrough).
2. Whether the requested model is a reasoning model that requires state repair.
3. The session key to use for cache lookups.

Detection is based on:
- `config.yaml` provider name
- The `model` field in the request body
- A built-in model capability registry (`aslor/models/registry.py`)

**Inputs:** request body dict, config  
**Outputs:** `DetectionResult(provider, model_name, needs_repair, session_key)`

---

### ReasoningRepairAgent

**File:** `aslor/reasoning/repair.py`

When `needs_repair=True`, this agent:

1. Looks up the session key in the `ReasoningStateStore`.
2. Finds assistant messages in the request that are missing `reasoning_content`
   (or `thinking` for Anthropic).
3. Re-injects the cached field into those messages.

If no cached state is found, the request is forwarded unchanged (safe
passthrough — never blocks a request).

**Inputs:** request messages list, session key, provider name  
**Outputs:** repaired messages list  
**Side effects:** read from `ReasoningStateStore`

---

### ProviderRouterAgent

**File:** `aslor/providers/registry.py`

Selects the correct `ProviderAdapter` instance based on `DetectionResult`.

In future versions, this agent will also do:
- cost-aware routing (cheap model for short requests)
- fallback routing (try provider A, fall back to B)
- task-based routing (architecture reviews → reasoning model)

**Inputs:** `DetectionResult`  
**Outputs:** `ProviderAdapter` instance

---

### UpstreamForwardAgent

**File:** `aslor/agents/forwarder.py`

Sends the (possibly repaired) request to the upstream LLM provider using the
selected adapter.

Handles:
- Setting correct headers (Authorization, Content-Type)
- Timeout management
- Provider-specific request normalization
- HTTP error handling with structured error responses

**Inputs:** repaired request dict, `ProviderAdapter`, config  
**Outputs:** raw upstream `httpx.Response`

---

### ResponseCaptureAgent

**File:** `aslor/agents/capture.py`

After a successful upstream response, this agent:

1. Parses the response (streaming or non-streaming).
2. Extracts reasoning state fields (`reasoning_content`, `thinking`).
3. Stores the full assistant message in `ReasoningStateStore` keyed by session.
4. Passes the original response through unmodified.

**Inputs:** upstream response, session key, provider name  
**Outputs:** same response (pass-through), side effect: cache write  
**Side effects:** write to `ReasoningStateStore`

---

### StreamRelayAgent

**File:** `aslor/agents/relay.py`

Streams SSE (Server-Sent Events) chunks from the upstream response back to
Android Studio.

For non-streaming responses, wraps the response in a single chunk.

Also handles:
- Detecting `reasoning_content` in streaming delta chunks (DeepSeek sends
  reasoning as a separate delta field).
- Accumulating streamed reasoning for the `ResponseCaptureAgent`.

**Inputs:** upstream response iterator  
**Outputs:** `StreamingResponse` (FastAPI)

---

## Support Modules (Not Agents)

| Module | Purpose |
|--------|---------|
| `aslor/cache/db.py` | Encrypted SQLite wrapper |
| `aslor/reasoning/state.py` | `ReasoningStateStore` CRUD |
| `aslor/config.py` | YAML config loader with env expansion |
| `aslor/logging_config.py` | Structured JSON logging |
| `aslor/models/request.py` | Pydantic request schemas |
| `aslor/models/response.py` | Pydantic response schemas |

---

## Provider Adapters

Each provider adapter implements the `ProviderAdapter` abstract base:

```python
class ProviderAdapter(ABC):
    def normalize_request(self, body: dict) -> dict: ...
    def is_reasoning_model(self, model: str) -> bool: ...
    def extract_reasoning_state(self, response: dict) -> dict | None: ...
    def inject_reasoning_state(self, messages: list, state: dict) -> list: ...
    def get_headers(self, api_key: str) -> dict: ...
    def get_base_url(self) -> str: ...
```

| Adapter | File | Reasoning Field |
|---------|------|-----------------|
| `OpenAIAdapter` | `providers/openai.py` | `reasoning_content` on assistant message |
| `DeepSeekAdapter` | `providers/deepseek.py` | `reasoning_content` on assistant message |
| `AnthropicAdapter` | `providers/anthropic.py` | `thinking` content block |
| `PassthroughAdapter` | `providers/passthrough.py` | none (safe forward) |

---

## Coupling Rules

To keep this service maintainable and testable, agents follow these rules:

1. **No agent imports another agent directly.** The pipeline wires them.
2. **Agents do not import from `server/`.** Server imports agents, not the reverse.
3. **Provider adapters do not import from `reasoning/`.** The pipeline handles the
   wiring between adapters and the reasoning store.
4. **Cache layer is only accessed through `ReasoningStateStore`.** No raw SQL
   outside `cache/db.py`.
5. **Config is injected, never imported as a global.** Agents receive config as a
   constructor argument or function parameter.

---

## Testing Each Agent

Every agent has a corresponding test module in `tests/`:

| Agent | Test file |
|-------|-----------|
| `ReasoningRepairAgent` | `tests/test_repair.py` |
| `ModelDetectorAgent` | `tests/test_detector.py` |
| `ProviderAdapters` | `tests/test_adapters.py` |
| `ResponseCaptureAgent` | `tests/test_capture.py` |
| `StreamRelayAgent` | `tests/test_relay.py` |
| `ReasoningStateStore` | `tests/test_cache.py` |
| Full pipeline routes | `tests/test_routes.py` |

Run all tests:

```powershell
pytest tests/ -v --tb=short
```

---

## Adding a New Provider

1. Create `aslor/providers/yourprovider.py` implementing `ProviderAdapter`.
2. Register it in `aslor/providers/registry.py`.
3. Add a model capability entry in `aslor/models/registry.py` for the model names.
4. Add a test in `tests/test_adapters.py`.
5. Document the reasoning field in this file.

No other files need to change.
