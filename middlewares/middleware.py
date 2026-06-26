"""HTTP middleware for the QueueStorm API.

Currently exposes a response-time middleware that stamps every response with
an ``X-Process-Time`` header. Register via ``register_middleware(app)``.
"""

import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class ResponseTimeMiddleware(BaseHTTPMiddleware):
    """Measure request handling time and expose it as ``X-Process-Time``."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        end_time = time.perf_counter()
        process_time = end_time - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


def register_middleware(app) -> None:
    """Attach ``ResponseTimeMiddleware`` to the given FastAPI app."""
    app.add_middleware(ResponseTimeMiddleware)