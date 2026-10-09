"""An in-call HTTP tool parameter with allowed values.

``enum`` on a parameter in ``in_call_tools.<name>.parameters`` reaches the
LLM as the parameter's JSON ``enum`` and is enforced before the request is
made, so the editor's enum type needs nothing more from the engine.
"""

import pytest

from src.tools.http.in_call_lookup import create_in_call_http_tool


def _tool():
    return create_in_call_http_tool(
        "book_slot",
        {
            "description": "Book a slot",
            "url": "https://api.example.com/book",
            "method": "POST",
            "parameters": [
                {
                    "name": "slot",
                    "type": "string",
                    "description": "Preferred part of the day",
                    "enum": ["morning", "afternoon", "evening"],
                    "required": True,
                },
                {"name": "note", "type": "string", "description": "Free text"},
            ],
        },
    )


def test_the_allowed_values_reach_the_openai_schema():
    schema = _tool().definition.to_openai_schema()

    properties = schema["function"]["parameters"]["properties"]
    assert properties["slot"] == {
        "type": "string",
        "description": "Preferred part of the day",
        "enum": ["morning", "afternoon", "evening"],
    }
    assert "enum" not in properties["note"]
    assert schema["function"]["parameters"]["required"] == ["slot"]


@pytest.mark.asyncio
async def test_a_value_outside_the_list_is_rejected_before_any_request():
    tool = _tool()

    assert await tool.validate_parameters({"slot": "evening"}) is True
    with pytest.raises(ValueError, match="Invalid value for slot. Must be one of: morning, afternoon, evening"):
        await tool.validate_parameters({"slot": "night"})
