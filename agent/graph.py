"""
LangGraph state machine assembly.

Wires the individual node functions into a directed graph:

    perceive -> reason -> act -> verify -> [correct -> act -> verify -> ...] -> END

The Correct node loops back to Act so the corrective tool call actually
executes. Max iterations are enforced inside Verify.
"""

from langgraph.graph import StateGraph, END

from agent.nodes import (
    AgentState,
    perceive,
    reason,
    act,
    verify,
    correct,
    route_after_verify,
)


def build_agent_graph():
    """Assemble and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("perceive", perceive)
    graph.add_node("reason", reason)
    graph.add_node("act", act)
    graph.add_node("verify", verify)
    graph.add_node("correct", correct)

    graph.set_entry_point("perceive")

    graph.add_edge("perceive", "reason")
    graph.add_edge("reason", "act")
    graph.add_edge("act", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "correct": "correct",
            "END": END,
        },
    )

    # Correct loops back to Act so the new tool call executes
    graph.add_edge("correct", "act")

    return graph.compile()


# ---------------------------------------------------------------------------
# Standalone test: verify the graph compiles and the topology is correct
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_agent_graph()
    print("[OK] Agent graph compiled successfully.")
    print()
    print("Graph topology:")
    print("  perceive -> reason -> act -> verify")
    print("  verify --(correction_needed)--> correct -> act (loop)")
    print("  verify --(else)--> END")
    print()
    print(f"[OK] Nodes: {list(app.get_graph().nodes.keys())}")