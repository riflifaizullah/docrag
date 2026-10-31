"""Checks that the MCP server starts and lists its tools. Needs no backend.

    python mcp_server/smoke_test.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import server  # noqa: E402

EXPECTED = {"list_documents", "search_passages", "ask_documents", "reindex_documents"}


def main() -> None:
    tools = {t.name: t for t in asyncio.run(server.server.list_tools())}
    assert set(tools) == EXPECTED, set(tools)
    assert tools["search_passages"].annotations.read_only_hint is True
    assert tools["reindex_documents"].annotations.read_only_hint is False
    print("ok:", ", ".join(sorted(tools)))


if __name__ == "__main__":
    main()
