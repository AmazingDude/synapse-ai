from tools.search import format_search_results, get_search_tool, web_search  # CHANGED: drop search_tool/tools (now lazy via get_search_tool)

__all__ = ["get_search_tool", "web_search", "format_search_results"]  # CHANGED
