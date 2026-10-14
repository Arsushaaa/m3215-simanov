from dataclasses import replace

import pytest
from aiohttp import web

from app.llm import ModelError, OpenRouterClient


@pytest.fixture
async def http_server(unused_tcp_port):
    state = {
        "status": 200,
        "body": {"choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}]},
        "requests": [],
    }

    async def completion(request):
        state["requests"].append((dict(request.headers), await request.json()))
        return web.json_response(
            state["body"], status=state["status"], headers=state.get("headers", {})
        )

    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}/api/v1", state
    finally:
        await runner.cleanup()


async def test_client_sends_roles_parameters_and_free_price_limit(http_server, bot_settings):
    url, state = http_server
    client = OpenRouterClient(replace(bot_settings, openrouter_base_url=url))
    await client.start()
    messages = [{"role": "system", "content": "Инструкция"}, {"role": "user", "content": "Вопрос"}]
    try:
        assert await client.complete(messages, 0.7) == "Ответ"
        headers, body = state["requests"][0]
        assert headers["Authorization"] == "Bearer fake-openrouter"
        assert body == {
            "model": bot_settings.openrouter_model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
            "reasoning": {"enabled": False},
            "provider": {"max_price": {"prompt": 0, "completion": 0}},
        }
    finally:
        await client.close()
    assert client.session.closed


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 302])
async def test_http_errors_are_safe_and_not_retried(http_server, bot_settings, status):
    url, state = http_server
    state.update(status=status, body={"error": "private key and prompt"})
    if status == 302:
        state["headers"] = {"Location": "https://untrusted.invalid/steal"}
    client = OpenRouterClient(replace(bot_settings, openrouter_base_url=url))
    await client.start()
    try:
        with pytest.raises(ModelError) as exc:
            await client.complete([{"role": "user", "content": "Вопрос"}], 0.3)
        assert "private" not in exc.value.user_message
        assert len(state["requests"]) == 1
    finally:
        await client.close()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {"content": []}}]},
    ],
)
async def test_empty_or_invalid_result_is_rejected(http_server, bot_settings, body):
    url, state = http_server
    state["body"] = body
    client = OpenRouterClient(replace(bot_settings, openrouter_base_url=url))
    await client.start()
    try:
        with pytest.raises(ModelError):
            await client.complete([{"role": "user", "content": "Вопрос"}], 0.3)
    finally:
        await client.close()


async def test_truncated_result_explains_model_limit_and_parts_without_retry(
    http_server, bot_settings
):
    url, state = http_server
    state["body"] = {
        "choices": [{"message": {"content": "Оборванный ответ"}, "finish_reason": "length"}]
    }
    client = OpenRouterClient(replace(bot_settings, openrouter_base_url=url))
    await client.start()
    try:
        with pytest.raises(ModelError, match="ограничении длины") as exc:
            await client.complete([{"role": "user", "content": "Эссе"}], 0.55)
        assert "по частям" in exc.value.user_message
        assert len(state["requests"]) == 1
    finally:
        await client.close()


async def test_timeout_is_safe(bot_settings, monkeypatch):
    client = OpenRouterClient(bot_settings)
    await client.start()

    def timeout(*args, **kwargs):
        raise TimeoutError("private request details")

    monkeypatch.setattr(client.session, "post", timeout)
    try:
        with pytest.raises(ModelError) as exc:
            await client.complete([{"role": "user", "content": "Вопрос"}], 0.3)
        assert exc.value.user_message == "Модель не успела ответить. Попробуй ещё раз позже."
    finally:
        await client.close()


async def test_network_error_is_safe(bot_settings, monkeypatch):
    from aiohttp import ClientConnectionError

    client = OpenRouterClient(bot_settings)
    await client.start()

    def unavailable(*args, **kwargs):
        raise ClientConnectionError("private connection details")

    monkeypatch.setattr(client.session, "post", unavailable)
    try:
        with pytest.raises(ModelError) as exc:
            await client.complete([{"role": "user", "content": "Вопрос"}], 0.3)
        assert exc.value.user_message == "Не удалось связаться с моделью. Попробуй позже."
    finally:
        await client.close()
