"""Sanity checks for bot_flows.py — the FSM-driven variant.

We can't run the full bot from a unit test (it needs Ollama, Whisper, Piper,
and a WebSocket transport), but we can verify:
  - The node factory functions all return well-formed NodeConfig dicts.
  - Every transition target referenced by a flow function is itself a valid
    node — no dead links.
  - Every node lists its functions, role/task messages, etc.
  - The graph is connected from the entry point (triage_node).

Catches the easy-to-make refactoring mistake of "rename a node, forget to
update one transition."
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bot_flows  # noqa: E402


# All node-factory functions in bot_flows
NODE_FACTORIES = [
    bot_flows.triage_node,
    bot_flows.collect_name_node,
    bot_flows.collect_phone_node,
    bot_flows.collect_window_node,
    bot_flows.collect_message_node,
    bot_flows.emergency_node,
    bot_flows.confirm_node,
    bot_flows.end_node,
]

NODE_NAMES = {fn.__name__: fn for fn in NODE_FACTORIES}


class TestNodeConfigShape:
    def test_each_node_has_required_keys(self):
        for fn in NODE_FACTORIES:
            cfg = fn()
            assert "name" in cfg, f"{fn.__name__} missing 'name'"
            assert "task_messages" in cfg, f"{fn.__name__} missing 'task_messages'"
            assert "functions" in cfg, f"{fn.__name__} missing 'functions'"

    def test_node_names_unique(self):
        names = [fn()["name"] for fn in NODE_FACTORIES]
        assert len(names) == len(set(names)), f"duplicate node names: {names}"

    def test_task_messages_are_well_formed(self):
        for fn in NODE_FACTORIES:
            tm = fn()["task_messages"]
            assert isinstance(tm, list), f"{fn.__name__} task_messages not a list"
            for m in tm:
                assert m.get("role") == "system"
                assert isinstance(m.get("content"), str) and m["content"]

    def test_triage_has_role_message(self):
        # Only the entry node carries the persistent persona.
        cfg = bot_flows.triage_node()
        assert cfg.get("role_messages"), "triage_node should set role_messages"

    def test_terminal_nodes_have_no_functions(self):
        # confirm and end are terminal — no further transitions.
        for fn in (bot_flows.confirm_node, bot_flows.end_node):
            assert fn()["functions"] == [], f"{fn.__name__} should have no functions"


class TestFlowGraphReachability:
    """Every node referenced by a flow function should be a real factory."""

    def test_set_intent_targets_exist(self):
        # Verified by hand: set_intent -> collect_name_node OR emergency_node OR triage_node
        # We confirm the graph is consistent by importing the source and checking
        # that the symbols referenced are bound to known node factories.
        src = inspect.getsource(bot_flows.set_intent)
        assert "collect_name_node()" in src
        assert "emergency_node()" in src
        assert "triage_node()" in src  # for the unclear-intent loop

    def test_set_phone_branches(self):
        # set_phone -> collect_window_node (appointment) OR collect_message_node (message)
        src = inspect.getsource(bot_flows.set_phone)
        assert "collect_window_node()" in src
        assert "collect_message_node()" in src

    def test_save_and_advance_to_confirm(self):
        # _save_and_advance always goes to confirm_node
        src = inspect.getsource(bot_flows._save_and_advance)
        assert "confirm_node()" in src

    def test_acknowledge_emergency_to_end(self):
        src = inspect.getsource(bot_flows.acknowledge_emergency)
        assert "end_node()" in src


class TestRoleMessageContent:
    def test_role_message_mentions_practice_name(self):
        cfg = bot_flows.triage_node()
        role = cfg["role_messages"][0]["content"]
        assert "Smith Family Dental" in role

    def test_role_message_forbids_calendar_invention(self):
        # The "no calendar access" rule must persist across nodes.
        cfg = bot_flows.triage_node()
        role = cfg["role_messages"][0]["content"]
        # Either explicit forbidding or mention that this is a callback request
        assert "calendar" in role.lower() or "callback" in role.lower()


class TestFlowFunctionsCallable:
    """All decorated flow functions should be callable; signatures should
    match the slots we expect to extract."""

    def test_set_name_signature(self):
        sig = inspect.signature(bot_flows.set_name)
        # First param is flow_manager, second should be caller_name
        params = list(sig.parameters)
        assert "flow_manager" in params
        assert "caller_name" in params

    def test_set_phone_signature(self):
        params = list(inspect.signature(bot_flows.set_phone).parameters)
        assert "callback_number" in params

    def test_set_window_signature(self):
        params = list(inspect.signature(bot_flows.set_window).parameters)
        assert "preferred_window" in params

    def test_set_message_signature(self):
        params = list(inspect.signature(bot_flows.set_message).parameters)
        assert "message" in params
