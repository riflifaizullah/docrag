"""DocRAG as an MCP server, so an AI assistant can search and ask your documents as tools.

    python mcp_server/server.py

It talks to a running DocRAG backend (DOCRAG_BACKEND_URL, default http://127.0.0.1:8000) and speaks MCP over
standard input and output, which is how Claude Desktop and most MCP clients start a local server.
"""
import functools
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

sys.path.insert(0, str(Path(__file__).parent))
import tools  # noqa: E402

server = MCPServer(
    "docrag",
    instructions=(
        "Tools for question answering over the user's own PDF documents. Use search_passages to see the exact "
        "passages and pages, ask_documents for a written answer with sources, and list_documents to see what is "
        "indexed. If ask_documents says the documents do not contain the answer, say so instead of guessing."
    ),
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def shows_backend_errors(fn):
    """A backend problem becomes a tool error whose message the assistant (and the user) can read."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except tools.BackendError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


@server.tool(annotations=READ_ONLY)
@shows_backend_errors
def list_documents() -> list[dict]:
    """List the indexed documents with their type, page count and size."""
    return tools.list_documents()


@server.tool(annotations=READ_ONLY)
@shows_backend_errors
def search_passages(question: str, limit: int = 5) -> dict:
    """Find the passages that best match a question, with document and page for each. Fast, no language model.

    The result also says whether DocRAG would try to answer this question (`would_answer`), which is false when
    nothing relevant was found.
    """
    return tools.search_passages(question, limit)


@server.tool(annotations=READ_ONLY)
@shows_backend_errors
def ask_documents(question: str) -> dict:
    """Answer a question using only the documents, citing the source files and pages.

    Slow on a CPU because a local language model writes the answer. It refuses when the documents do not cover
    the question, and may ask which document is meant (needs_clarification, candidate_documents).
    """
    return tools.ask_documents(question)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
@shows_backend_errors
def reindex_documents() -> dict:
    """Re-read the documents folder: index new and changed files and drop removed ones. Changes the index."""
    return tools.reindex_documents()


if __name__ == "__main__":
    server.run()
