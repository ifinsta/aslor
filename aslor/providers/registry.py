"""Provider adapter selection — driven by ``models.yaml`` registry."""

from __future__ import annotations

from aslor.config import AppConfig
from aslor.models.registry import get_registry
from aslor.providers.base import ProviderAdapter


def get_adapter(config: AppConfig, model_name: str = "") -> ProviderAdapter:
    """Return the appropriate :class:`ProviderAdapter` for the current config.

    Decision order:
    1. Look up ``config.provider.name`` in the model registry, use its
       ``adapter`` field.
    2. If provider name is unknown, try to find which provider owns
       *model_name* and use its adapter.
    3. Fallback to :class:`PassthroughAdapter`.
    """
    from aslor.providers.anthropic import AnthropicAdapter
    from aslor.providers.chutes import ChutesAdapter
    from aslor.providers.deepseek import DeepSeekAdapter
    from aslor.providers.openai import OpenAIAdapter
    from aslor.providers.passthrough import PassthroughAdapter

    _CLASSES: dict[str, type[ProviderAdapter]] = {
        "openai": OpenAIAdapter,
        "deepseek": DeepSeekAdapter,
        "anthropic": AnthropicAdapter,
        "chutes": ChutesAdapter,
        "passthrough": PassthroughAdapter,
    }

    reg = get_registry()
    provider_name = config.provider.name.lower()
    base_url = config.provider.base_url

    # Prefer model-specific provider hints when the configured provider is a
    # generic passthrough. This keeps reasoning-aware adapters active even when
    # the upstream endpoint is an OpenAI-compatible proxy.
    model_provider = reg.get_provider_for_model(model_name) if model_name else None
    provider = reg.get_provider(provider_name)
    if provider_name in {"", "passthrough"} and model_provider is not None:
        provider = model_provider
    elif provider is None and model_provider is not None:
        provider = model_provider

    adapter_name = provider.adapter.lower() if provider else provider_name

    if adapter_name in _CLASSES:
        cls = _CLASSES[adapter_name]
        if adapter_name == "anthropic":
            return cls(
                base_url=base_url,
                thinking_enabled=config.provider.thinking_enabled,
                thinking_budget_tokens=config.provider.thinking_budget_tokens,
            )
        return cls(base_url=base_url)

    return PassthroughAdapter(base_url=base_url)
