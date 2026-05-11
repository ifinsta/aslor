"""UpstreamForwardAgent — sends the repaired request to the upstream provider."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

from aslor.config import AppConfig
from aslor.providers.base import ProviderAdapter

logger = logging.getLogger(__name__)


async def forward(
    body: dict[str, Any],
    adapter: ProviderAdapter,
    config: AppConfig,
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    """Send *body* to the upstream provider and return the raw response.

    If *api_key* is provided (extracted from the incoming request), it takes
    precedence over the configured key.  The response is returned with
    ``stream=True`` so callers can iterate over SSE chunks or read the full
    body as needed.
    """
    url = f"{adapter.get_base_url()}/chat/completions"
    headers = adapter.get_headers(api_key if api_key else config.provider.api_key)
    timeout = config.provider.timeout_seconds

    upstream = client or httpx.AsyncClient(timeout=timeout)
    response = await upstream.send(upstream.build_request("POST", url, headers=headers, json=body), stream=True)
    return response
