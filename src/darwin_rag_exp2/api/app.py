"""FastAPI application for the Exp2 message endpoint."""

from __future__ import annotations

from pydantic import BaseModel

from .runtime import build_default_message_service
from .service import MessageService, MessageValidationError


class MessageRequest(BaseModel):
    query: str


class MessageResponse(BaseModel):
    answer: str


def create_app(*, service: MessageService | None = None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="DARWIN-RAG Exp2 API")
    app.state.message_service = service

    def handle_message(payload: MessageRequest) -> MessageResponse:
        active_service = _get_message_service(app)
        try:
            answer = active_service.answer(payload.query)
        except MessageValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return MessageResponse(answer=answer)

    app.post("/api/messages/", response_model=MessageResponse)(handle_message)
    app.post("/api/messages", response_model=MessageResponse, include_in_schema=False)(
        handle_message
    )
    return app


def _get_message_service(app) -> MessageService:
    service = app.state.message_service
    if service is None:
        service = build_default_message_service()
        app.state.message_service = service
    return service


app = create_app()
