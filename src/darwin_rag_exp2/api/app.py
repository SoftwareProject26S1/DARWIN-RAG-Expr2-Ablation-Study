"""FastAPI application for the Exp2 message endpoint."""

from __future__ import annotations

import logging
from time import perf_counter

from pydantic import BaseModel

from .runtime import build_default_message_service
from .service import MessageService, MessageValidationError


logger = logging.getLogger(__name__)


class MessageRequest(BaseModel):
    query: str


class MessageResponse(BaseModel):
    answer: str


def create_app(*, service: MessageService | None = None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="DARWIN-RAG Exp2 API")
    app.state.message_service = service

    def handle_message(payload: MessageRequest) -> MessageResponse:
        started_at = perf_counter()
        logger.info("message request started query_len=%s", len(payload.query))
        active_service = _get_message_service(app)
        try:
            answer = active_service.answer(payload.query)
        except MessageValidationError as error:
            logger.warning(
                "message request rejected query_len=%s detail=%s",
                len(payload.query),
                error,
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception:
            logger.exception("message request failed query_len=%s", len(payload.query))
            raise
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        logger.info(
            "message request completed query_len=%s answer_len=%s elapsed_ms=%.1f",
            len(payload.query),
            len(answer),
            elapsed_ms,
        )
        return MessageResponse(answer=answer)

    app.post("/api/messages/", response_model=MessageResponse)(handle_message)
    app.post("/api/messages", response_model=MessageResponse, include_in_schema=False)(
        handle_message
    )
    return app


def _get_message_service(app) -> MessageService:
    service = app.state.message_service
    if service is None:
        logger.info("default message service build started")
        try:
            service = build_default_message_service()
        except Exception:
            logger.exception("default message service build failed")
            raise
        app.state.message_service = service
        logger.info("default message service build completed")
    return service


app = create_app()
