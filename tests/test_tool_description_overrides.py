"""Built-in tools described to the LLM in the operator's own words.

The function schema is all a pipeline's LLM learns about a tool, and its
text was fixed in the code. ``tools.<name>.description`` and
``tools.<name>.parameter_descriptions`` now replace that text before any
schema is built; execution stays the tool's own.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.config import AppConfig, OpenAIProviderConfig
from src.pipelines.openai import OpenAILLMAdapter
from src.tools.base import DescribedTool
from src.tools.registry import ToolRegistry
from src.tools.runtime_config import ToolRuntimeGeneration
from src.tools.telephony.hangup import HangupCallTool

DESCRIPTION = (
    "Завершить звонок. Вызывай, когда абонент попрощался или подтвердил, что вопросов больше нет. "
    "Прощальную фразу передай в farewell_message, а текст ответа оставь пустым."
)
PARAMETER = "Прощальная фраза, которую агент произнесёт перед завершением звонка."


def _generation(tools_config):
    return ToolRuntimeGeneration.build(generation_id=1, config={"tools": tools_config})


def _hangup_config(**extra):
    return {"hangup_call": {"enabled": True, "farewell_message": "Bye!", **extra}}


def test_the_configured_texts_replace_the_built_in_ones():
    generation = _generation(
        _hangup_config(description=DESCRIPTION, parameter_descriptions={"farewell_message": PARAMETER})
    )

    tool = generation.registry.get("hangup_call")
    definition = tool.definition

    assert isinstance(tool, DescribedTool)
    assert definition.description == DESCRIPTION
    assert [p.description for p in definition.parameters] == [PARAMETER]
    # Everything else is the built-in definition.
    built_in = HangupCallTool().definition
    assert definition.name == built_in.name
    assert definition.category == built_in.category
    assert definition.phase == built_in.phase
    assert definition.requires_channel == built_in.requires_channel
    assert [(p.name, p.type, p.required) for p in definition.parameters] == [
        (p.name, p.type, p.required) for p in built_in.parameters
    ]
    schema = definition.to_openai_schema()
    assert schema["function"]["description"] == DESCRIPTION
    assert schema["function"]["parameters"]["properties"]["farewell_message"]["description"] == PARAMETER


def test_either_text_can_be_configured_on_its_own():
    only_description = _generation(_hangup_config(description=DESCRIPTION)).registry.get("hangup_call").definition
    assert only_description.description == DESCRIPTION
    assert only_description.parameters[0].description == HangupCallTool().definition.parameters[0].description

    only_parameter = (
        _generation(_hangup_config(parameter_descriptions={"farewell_message": PARAMETER}))
        .registry.get("hangup_call")
        .definition
    )
    assert only_parameter.description == HangupCallTool().definition.description
    assert only_parameter.parameters[0].description == PARAMETER


def test_blank_texts_keep_the_built_in_tool_untouched():
    generation = _generation(
        _hangup_config(description="   ", parameter_descriptions={"farewell_message": ""})
    )
    tool = generation.registry.get("hangup_call")
    assert type(tool) is HangupCallTool
    assert tool.definition.description == HangupCallTool().definition.description


def test_unknown_parameters_and_unknown_tools_are_ignored(caplog):
    generation = _generation(
        {
            **_hangup_config(
                description=DESCRIPTION,
                parameter_descriptions={"farewell_message": PARAMETER, "no_such_parameter": "x"},
            ),
            "no_such_tool": {"description": "ignored"},
            "transfer": {"technology": "SIP"},
        }
    )

    definition = generation.registry.get("hangup_call").definition
    assert definition.description == DESCRIPTION
    assert [p.name for p in definition.parameters] == ["farewell_message"]
    assert generation.registry.get("no_such_tool") is None
    assert any("no_such_parameter" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_execution_is_still_the_built_in_tools():
    generation = _generation(_hangup_config(description=DESCRIPTION))
    tool = generation.registry.get("hangup_call")
    context = SimpleNamespace(
        call_id="call-1",
        update_session=AsyncMock(),
        get_config_value=lambda key, default=None: default,
    )

    assert await tool.validate_parameters({"farewell_message": "До свидания!"}) is True
    result = await tool.execute({"farewell_message": "До свидания!"}, context)

    assert result["will_hangup"] is True
    assert result["farewell_message"] == "До свидания!"
    context.update_session.assert_awaited_once_with(cleanup_after_tts=True)
    assert isinstance(tool.wrapped, HangupCallTool)


def test_the_openai_adapter_advertises_the_configured_text():
    generation = _generation(
        _hangup_config(description=DESCRIPTION, parameter_descriptions={"farewell_message": PARAMETER})
    )
    app_config = AppConfig(
        default_provider="native_llm",
        providers={"native_llm": {"api_key": "test-key"}},
        asterisk={"host": "127.0.0.1", "username": "ari", "password": "secret"},
        llm={"initial_greeting": "hi", "prompt": "persona", "model": "gpt-4o"},
        audio_transport="audiosocket",
        downstream_mode="stream",
    )
    adapter = OpenAILLMAdapter("native_llm", app_config, OpenAIProviderConfig(api_key="test-key"), {})
    adapter.bind_tool_registry(generation.registry)

    payload = {}
    adapter._attach_tools(payload, {"tools": ["hangup_call"]})

    function = payload["tools"][0]["function"]
    assert function["name"] == "hangup_call"
    assert function["description"] == DESCRIPTION
    assert function["parameters"]["properties"]["farewell_message"]["description"] == PARAMETER
    assert payload["tool_choice"] == "auto"


def test_a_reload_re_describes_without_touching_the_earlier_generation():
    first = _generation(_hangup_config(description="Первая формулировка."))
    second = ToolRuntimeGeneration.build(
        generation_id=2, config={"tools": _hangup_config(description="Вторая формулировка.")}
    )

    assert first.registry.get("hangup_call").definition.description == "Первая формулировка."
    assert second.registry.get("hangup_call").definition.description == "Вторая формулировка."
    # Cloned registries (per-agent inline HTTP tools) keep the wording.
    assert second.registry.clone().get("hangup_call").definition.description == "Вторая формулировка."


def test_applying_twice_wraps_the_built_in_tool_once():
    registry = ToolRegistry.isolated()
    registry.initialize_default_tools()
    registry.apply_definition_overrides(_hangup_config(description="Одна."))
    registry.apply_definition_overrides(_hangup_config(description="Другая."))

    tool = registry.get("hangup_call")
    assert tool.definition.description == "Другая."
    assert type(tool.wrapped) is HangupCallTool
