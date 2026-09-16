"""
Hangup Call Tool - End the current call.

Allows full AI agents to end calls when appropriate (e.g., after goodbye).

Simplified design (v5.0):
- Trust the AI to manage conversation flow via system prompt
- No complex guardrails that cause race conditions
- Just: farewell message → mark for cleanup → hangup after audio

With ``tools.hangup_call.farewell_message_enabled: false`` the tool has no
``farewell_message`` parameter at all: the model says goodbye in its reply,
the tool only marks the call, and the engine ends it once that reply has
been heard.
"""

from typing import Any, Dict, Mapping
from src.tools.base import Tool, ToolDefinition, ToolParameter, ToolCategory
from src.tools.context import ToolExecutionContext
import structlog

logger = structlog.get_logger(__name__)

# The texts the LLM sees, unless tools.hangup_call.description overrides them.
DESCRIPTION = (
    "End the current call. Call this when the caller says goodbye or thank you and is ready to hang up. "
    "Set farewell_message to your goodbye sentence."
)
DESCRIPTION_WITHOUT_FAREWELL = (
    "End the current call. Call this when the caller says goodbye or thank you and is ready to hang up, "
    "in the same reply as your own goodbye: the call ends once your reply has been spoken."
)
FAREWELL_PARAMETER_DESCRIPTION = "Farewell message to speak before hanging up. Should be warm and professional."


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


class HangupCallTool(Tool):
    """
    End the current call.

    Use when:
    - Caller says goodbye/thank you/that's all
    - Call purpose is complete
    - Caller explicitly asks to end the call

    Only available to full agents (not partial/assistant agents).
    """

    def __init__(self) -> None:
        self._farewell_message_enabled = True

    def configure(self, config: Mapping[str, Any]) -> None:
        """Take the ``tools.hangup_call`` block: ``farewell_message_enabled`` (default true)."""
        self._farewell_message_enabled = _as_bool(
            (config or {}).get("farewell_message_enabled"), True
        )

    @property
    def farewell_message_enabled(self) -> bool:
        return self._farewell_message_enabled

    @property
    def definition(self) -> ToolDefinition:
        if not self._farewell_message_enabled:
            return ToolDefinition(
                name="hangup_call",
                description=DESCRIPTION_WITHOUT_FAREWELL,
                category=ToolCategory.TELEPHONY,
                requires_channel=True,
                max_execution_time=5,
                parameters=[],
            )
        return ToolDefinition(
            name="hangup_call",
            description=DESCRIPTION,
            category=ToolCategory.TELEPHONY,
            requires_channel=True,
            max_execution_time=5,
            parameters=[
                ToolParameter(
                    name="farewell_message",
                    type="string",
                    description=FAREWELL_PARAMETER_DESCRIPTION,
                    required=False
                )
            ]
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> Dict[str, Any]:
        """
        End the call.

        Simplified v5.0 design:
        - Get farewell message (from parameter or config default)
        - Mark session for cleanup after TTS
        - Return success with will_hangup flag

        With the farewell disabled there is no farewell at all: the result
        carries an empty message and the engine ends the call once the reply
        that asked for the hangup has been heard.

        The AI manages transcript offers via system prompt - no guardrails needed.

        Args:
            parameters: {farewell_message: Optional[str]}
            context: Tool execution context

        Returns:
            {
                status: "success" | "error",
                message: "Farewell message" (empty when the farewell is disabled),
                will_hangup: true
            }
        """
        if not self._farewell_message_enabled:
            logger.info(
                "📞 Hangup requested (farewell_message disabled; the reply is the goodbye)",
                call_id=context.call_id,
            )
            try:
                await context.update_session(cleanup_after_tts=True)
                return {
                    "status": "success",
                    "message": "",
                    "farewell_message": "",
                    "will_hangup": True,
                }
            except Exception as e:
                logger.error(f"Error preparing hangup: {e}", exc_info=True)
                return {
                    "status": "error",
                    "message": "",
                    "farewell_message": "",
                    "will_hangup": True,
                    "error": str(e),
                }

        farewell = parameters.get('farewell_message')

        if not farewell:
            farewell = context.get_config_value(
                'tools.hangup_call.farewell_message',
                "Thank you for calling. Goodbye!"
            )

        logger.info("📞 Hangup requested",
                   call_id=context.call_id,
                   farewell=farewell)

        try:
            # Mark the session so the engine will hang up after the farewell audio finishes.
            await context.update_session(cleanup_after_tts=True)
            logger.info("✅ Call will hangup after farewell", call_id=context.call_id)

            return {
                "status": "success",
                "message": farewell,
                # Canonical terminal-audio field for provider adapters. Keep
                # `message` for backwards compatibility with generic tools.
                "farewell_message": farewell,
                "will_hangup": True
            }

        except Exception as e:
            logger.error(f"Error preparing hangup: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "Goodbye!",
                "farewell_message": "Goodbye!",
                "will_hangup": True,
                "error": str(e)
            }
