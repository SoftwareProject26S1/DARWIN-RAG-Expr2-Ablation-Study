class FakeMessageService:
    def __init__(self):
        self.queries = []

    def answer(self, query):
        self.queries.append(query)
        if not query.strip():
            from darwin_rag_exp2.api.service import MessageValidationError

            raise MessageValidationError("query must not be blank")
        return "API 답변입니다."


def _client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)


def test_post_api_messages_returns_answer_body_only():
    from darwin_rag_exp2.api.app import create_app

    service = FakeMessageService()
    client = _client(create_app(service=service))

    response = client.post("/api/messages/", json={"query": "수강신청 기간은?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "API 답변입니다."}
    assert service.queries == ["수강신청 기간은?"]


def test_post_api_messages_accepts_no_slash_alias():
    from darwin_rag_exp2.api.app import create_app

    client = _client(create_app(service=FakeMessageService()))

    response = client.post("/api/messages", json={"query": "장학 일정은?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "API 답변입니다."}


def test_post_api_messages_rejects_missing_query():
    from darwin_rag_exp2.api.app import create_app

    client = _client(create_app(service=FakeMessageService()))

    response = client.post("/api/messages/", json={})

    assert response.status_code == 422


def test_post_api_messages_maps_blank_query_to_400():
    from darwin_rag_exp2.api.app import create_app

    client = _client(create_app(service=FakeMessageService()))

    response = client.post("/api/messages/", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "query must not be blank"


def test_api_messages_allows_vite_localhost_cors_preflight():
    from darwin_rag_exp2.api.app import create_app

    client = _client(create_app(service=FakeMessageService()))

    response = client.options(
        "/api/messages",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_api_messages_reads_allowed_cors_origins_from_env(monkeypatch):
    from darwin_rag_exp2.api.app import create_app

    monkeypatch.setenv(
        "DARWIN_EXP2_API_CORS_ALLOWED_ORIGINS",
        "http://localhost:5173, http://127.0.0.1:5173",
    )
    client = _client(create_app(service=FakeMessageService()))

    response = client.options(
        "/api/messages",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
