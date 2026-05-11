"""Request ID + structured logging middleware."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from aslor.server.log_buffer import push as log_push

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.time()
        msg = f"req {request_id} {request.method} {request.url.path}"
        logger.info(msg)
        log_push("INFO", msg)
        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000)
        res_msg = f"res {request_id} {response.status_code} {elapsed_ms}ms"
        logger.info(res_msg)
        log_push("INFO", res_msg)
        return response
