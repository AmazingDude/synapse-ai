"""LangGraph package for the business research assistant.

Kept intentionally empty so that ``from graph.state import GraphState`` does
NOT trigger eager loading of the LLM and Tavily clients (which would happen
if ``build_graph`` were re-exported here).  Import what you need directly:

    from graph.state import GraphState
    from graph.graph_builder import build_graph
"""
