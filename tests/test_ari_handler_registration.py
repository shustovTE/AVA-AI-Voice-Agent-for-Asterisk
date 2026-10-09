"""ARI event handlers register once.

Engine.start() runs again on a reconnect or a second bootstrap path, and a
stacked handler made every ARI event fire its side effects twice. For
PlaybackFinished the second pass found the playback already popped and warned
about an unknown id, right after the first had cleared the caller's gating.
"""
from unittest.mock import MagicMock, patch

from src.ari_client import ARIClient


def _client():
    with patch.object(ARIClient, "__init__", lambda self, *a, **k: None):
        client = ARIClient()
    client.event_handlers = {}
    return client


async def _handler(event):
    return None


async def _other_handler(event):
    return None


def test_the_same_handler_registers_once():
    client = _client()

    client.add_event_handler("PlaybackFinished", _handler)
    client.add_event_handler("PlaybackFinished", _handler)
    client.add_event_handler("PlaybackFinished", _handler)

    assert client.event_handlers["PlaybackFinished"] == [_handler]


def test_different_handlers_both_register():
    client = _client()

    client.add_event_handler("PlaybackFinished", _handler)
    client.add_event_handler("PlaybackFinished", _other_handler)

    assert client.event_handlers["PlaybackFinished"] == [_handler, _other_handler]


def test_the_same_handler_registers_per_event_type():
    client = _client()

    client.add_event_handler("PlaybackFinished", _handler)
    client.add_event_handler("ChannelTalkingStarted", _handler)

    assert client.event_handlers["PlaybackFinished"] == [_handler]
    assert client.event_handlers["ChannelTalkingStarted"] == [_handler]


def test_bound_methods_of_the_same_object_deduplicate():
    """Engine.start() re-registering self._on_playback_finished is the real case."""
    client = _client()
    engine = MagicMock()

    client.add_event_handler("PlaybackFinished", engine.handler)
    client.add_event_handler("PlaybackFinished", engine.handler)

    assert client.event_handlers["PlaybackFinished"] == [engine.handler]


def test_on_event_alias_deduplicates_too():
    client = _client()

    client.on_event("StasisStart", _handler)
    client.add_event_handler("StasisStart", _handler)

    assert client.event_handlers["StasisStart"] == [_handler]
