from __future__ import annotations

from mas.mas import AgentSystemManager


def _set_test_policy(manager: AgentSystemManager) -> None:
    manager._configure_model_failure_policy(
        {
            "base_cooldown_seconds": 60,
            "min_cooldown_seconds": 1,
            "max_cooldown_seconds": 120,
            "failure_half_life_seconds": 600,
            "history_retention_seconds": 3600,
        }
    )


def test_model_failure_reorders_until_cooldown_expires(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    now = [1_000.0]
    manager._model_health_now = lambda: now[0]
    _set_test_policy(manager)

    models = [
        {"provider": "groq", "model": "primary"},
        {"provider": "openrouter", "model": "fallback"},
    ]
    manager._register_model_infos(models)

    assert [m["model"] for m in manager.order_models_for_availability(models)] == [
        "primary",
        "fallback",
    ]

    manager.record_model_failure(
        "groq",
        "primary",
        models[0],
        {
            "ok": False,
            "status_code": 500,
            "errors": [{"type": "HTTPError", "message": "server error"}],
        },
    )

    assert [m["model"] for m in manager.order_models_for_availability(models)] == [
        "fallback",
        "primary",
    ]
    health = manager.get_model_health(model_info=models[0])
    assert health["in_cooldown"] is True
    assert health["consecutive_failures"] == 1
    assert health["last_error"]["type"] == "server"

    now[0] += 61
    assert [m["model"] for m in manager.order_models_for_availability(models)] == [
        "primary",
        "fallback",
    ]

    manager.record_model_success("groq", "primary", models[0], {"ok": True})
    health = manager.get_model_health(model_info=models[0])
    assert health["in_cooldown"] is False
    assert health["consecutive_failures"] == 0
    assert health["last_error"] is None


def test_repeated_failures_extend_cooldown_without_exceeding_cap(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    now = [2_000.0]
    manager._model_health_now = lambda: now[0]
    _set_test_policy(manager)

    model_info = {"provider": "groq", "model": "primary"}
    for _ in range(5):
        manager.record_model_failure(
            "groq",
            "primary",
            model_info,
            {
                "ok": False,
                "status_code": 429,
                "errors": [{"type": "HTTPError", "message": "rate limit"}],
            },
        )

    health = manager.get_model_health(model_info=model_info)
    assert health["consecutive_failures"] == 5
    assert health["cooldown_seconds"] > 60
    assert health["cooldown_seconds"] <= 120
    assert health["last_error"]["type"] == "rate_limit"


def test_agent_skips_cooled_model_and_records_synthetic_attempt(monkeypatch, workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    manager.get_key = lambda provider: "test-key"
    now = [3_000.0]
    manager._model_health_now = lambda: now[0]
    _set_test_policy(manager)
    manager.create_agent(
        name="agent",
        system="system",
        required_outputs={"response": "text"},
        models=[
            {"provider": "groq", "model": "primary"},
            {"provider": "groq", "model": "fallback"},
        ],
    )
    agent = manager.agents["agent"]
    agent._build_default_conversation = lambda messages, verbose: [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    manager.record_model_failure(
        "groq",
        "primary",
        agent.models[0],
        {
            "ok": False,
            "status_code": 500,
            "errors": [{"type": "HTTPError", "message": "server error"}],
        },
    )

    called_models = []

    def provider(**kwargs):
        model_name = kwargs["model_name"]
        called_models.append(model_name)
        if model_name == "primary":
            raise AssertionError("cooled model should not be called")
        content = '{"response":"fallback ok"}'
        agent._last_provider_response_metadata = agent._provider_success_metadata(
            "groq",
            model_name,
            raw_response={"id": "chatcmpl-backoff"},
            content=content,
            usage={"prompt_tokens": 2, "completion_tokens": 1},
        )
        return content

    monkeypatch.setattr(agent, "_call_groq_api", provider)

    result = agent.run()
    metadata = result[0]["metadata"]

    assert called_models == ["fallback"]
    assert result[0]["content"] == {"response": "fallback ok"}
    assert [attempt["model"] for attempt in metadata["provider_attempts"]] == [
        "primary",
        "fallback",
    ]
    skipped = metadata["provider_attempts"][0]
    assert skipped["skipped"] is True
    assert skipped["errors"][0]["type"] == "model_temporarily_suppressed"
    assert skipped["model_health"]["in_cooldown"] is True
    assert metadata["provider_response"]["model"] == "fallback"


def test_all_cooled_models_still_selects_one_attempt(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    now = [4_000.0]
    manager._model_health_now = lambda: now[0]
    _set_test_policy(manager)

    models = [
        {"provider": "groq", "model": "primary"},
        {"provider": "openrouter", "model": "secondary"},
    ]
    manager.record_model_failure(
        "groq",
        "primary",
        models[0],
        {
            "ok": False,
            "status_code": 500,
            "errors": [{"type": "HTTPError", "message": "server error"}],
        },
    )
    now[0] += 10
    manager.record_model_failure(
        "openrouter",
        "secondary",
        models[1],
        {
            "ok": False,
            "status_code": 500,
            "errors": [{"type": "HTTPError", "message": "server error"}],
        },
    )

    prepared = manager.prepare_model_attempts(models)

    assert prepared[0]["model_info"]["model"] == "primary"
    assert prepared[0]["skip"] is False
    assert prepared[1]["model_info"]["model"] == "secondary"
    assert prepared[1]["skip"] is True


def test_model_failure_state_is_manager_wide_across_agents(workspace_tmp_path):
    manager = AgentSystemManager(base_directory=str(workspace_tmp_path))
    now = [5_000.0]
    manager._model_health_now = lambda: now[0]
    _set_test_policy(manager)
    shared_failed_model = {"provider": "groq", "model": "shared-primary"}

    manager.create_agent(
        name="agent_one",
        system="system",
        required_outputs={"response": "text"},
        models=[shared_failed_model, {"provider": "openrouter", "model": "fallback-one"}],
    )
    manager.create_agent(
        name="agent_two",
        system="system",
        required_outputs={"response": "text"},
        models=[shared_failed_model, {"provider": "openrouter", "model": "fallback-two"}],
    )

    manager.record_model_failure(
        "groq",
        "shared-primary",
        manager.agents["agent_one"].models[0],
        {
            "ok": False,
            "status_code": 429,
            "errors": [{"type": "HTTPError", "message": "rate limit"}],
        },
    )

    prepared = manager.prepare_model_attempts(manager.agents["agent_two"].models)

    assert prepared[0]["model_info"]["model"] == "shared-primary"
    assert prepared[0]["skip"] is True
    assert prepared[0]["status"]["last_error"]["type"] == "rate_limit"
    assert prepared[1]["model_info"]["model"] == "fallback-two"
    assert prepared[1]["skip"] is False


def test_model_failure_policy_can_disable_suppression(workspace_tmp_path):
    manager = AgentSystemManager(
        base_directory=str(workspace_tmp_path),
        model_failure_policy=False,
    )
    now = [6_000.0]
    manager._model_health_now = lambda: now[0]
    models = [
        {"provider": "groq", "model": "primary"},
        {"provider": "openrouter", "model": "fallback"},
    ]

    manager.record_model_failure(
        "groq",
        "primary",
        models[0],
        {
            "ok": False,
            "status_code": 429,
            "errors": [{"type": "HTTPError", "message": "rate limit"}],
        },
    )

    prepared = manager.prepare_model_attempts(models)

    assert [entry["model_info"]["model"] for entry in prepared] == ["primary", "fallback"]
    assert [entry["skip"] for entry in prepared] == [False, False]
