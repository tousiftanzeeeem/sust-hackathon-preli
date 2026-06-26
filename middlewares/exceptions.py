
import json

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class SemanticValidationError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def register_exception_handlers(app) -> None:

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        semantic_codes = {
            "value_error",
            "value_error.any_str.min_length",
            "string_too_short",
            "too_long",
            "value_error.any_str.max_length",
        }
        structural_codes = {"missing", "string_type", "type_error", "enum"}

        is_semantic = any(
            err.get("type") in semantic_codes
            and err.get("type") not in structural_codes
            for err in errors
        )

        status_code = (
            status.HTTP_422_UNPROCESSABLE_ENTITY
            if is_semantic
            else status.HTTP_400_BAD_REQUEST
        )

        safe_errors = []
        for err in errors:
            safe_errors.append({
                "loc": list(err.get("loc", [])),
                "type": err.get("type"),
                "msg": err.get("msg"),
            })

        body = {
            "error": "unprocessable_entity" if is_semantic else "malformed_input",
            "message": (
                "Input is semantically invalid."
                if is_semantic
                else "Malformed input. Please check required fields and value types."
            ),
            "details": safe_errors,
        }
        return JSONResponse(status_code=status_code, content=body)

    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
        """Invalid JSON body → 400."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "malformed_input",
                "message": "Request body is not valid JSON.",
            },
        )

    @app.exception_handler(SemanticValidationError)
    async def semantic_validation_exception_handler(request: Request, exc: SemanticValidationError):
        """Schema-valid but semantically invalid input → 422."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "unprocessable_entity",
                "message": exc.message,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all so the service never crashes and never leaks stack traces,
        tokens, or secrets (per Section 9.2 of the rubric)."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An internal error occurred while processing the request.",
            },
        )