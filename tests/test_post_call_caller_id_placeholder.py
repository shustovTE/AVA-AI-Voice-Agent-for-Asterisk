"""The post-call tools name the extension an outbound call was placed from.

The identity (the lead's Caller ID override or the global outbound extension)
was computed at originate, logged and forgotten; ``{caller_number}`` and
``{called_number}`` both carry the lead's number, so a webhook could not say
which extension dialed the lead. ``{caller_id}`` and ``{caller_id_source}``
now carry it, in the body, the URL and the headers.
"""

from src.tools.context import PostCallContext
from src.tools.http.generic_webhook import GenericWebhookTool, WebhookConfig


def _tool() -> GenericWebhookTool:
    return GenericWebhookTool(
        WebhookConfig(
            name="crm",
            url="https://crm.example/{caller_id}/{caller_id_source}",
            payload_template='{"ext": "{caller_id}", "from": "{caller_id_source}"}',
        )
    )


def test_an_outbound_call_reports_the_identity_it_was_placed_from():
    context = PostCallContext(
        call_id="chan-1",
        caller_number="+74951234567",
        call_direction="outbound",
        caller_id="101",
        caller_id_source="lead",
    )
    payload = context.to_payload_dict()
    assert (payload["caller_id"], payload["caller_id_source"]) == ("101", "lead")
    tool = _tool()
    assert tool._substitute_variables(tool.config.url, context) == "https://crm.example/101/lead"
    assert tool._build_payload(context) == '{"ext": "101", "from": "lead"}'


def test_an_inbound_call_leaves_both_empty():
    context = PostCallContext(call_id="chan-2", caller_number="+74950000000")
    payload = context.to_payload_dict()
    assert (payload["caller_id"], payload["caller_id_source"]) == ("", "")
    tool = _tool()
    assert tool._substitute_variables(tool.config.url, context) == "https://crm.example//"
