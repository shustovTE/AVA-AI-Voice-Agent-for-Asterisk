"""Checks the engine runs on request from the Admin UI, where the calls run.

``providers`` probes a provider block the way a call would reach it;
``voices`` registers a reference voice on a self-hosted speech endpoint from
a file next to the engine. Both are importable by the Admin UI backend as
well, which runs them itself only when the engine cannot be reached.
"""
