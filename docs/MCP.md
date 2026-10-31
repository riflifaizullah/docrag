# MCP server

DocRAG can run as an [MCP](https://modelcontextprotocol.io) server, so an AI assistant such as Claude Desktop can search and ask your documents as tools. The server is a thin layer: it calls the DocRAG backend over HTTP, so the documents, the index and the language model stay where they are.

## Tools

| Tool | What it does | Changes anything? |
|---|---|---|
| `list_documents` | The indexed documents with type, page count and size | No |
| `search_passages` | The best matching passages with document and page, and whether DocRAG would try to answer (`would_answer`). No language model, so it is fast | No |
| `ask_documents` | A written answer grounded in the documents, with sources. It refuses when the documents do not cover the question, and may ask which document is meant. Slow on a CPU | No |
| `reindex_documents` | Reads the documents folder again and indexes new and changed files | Yes, the index |

`search_passages` is backed by a new backend endpoint, `POST /search`, described in [API.md](API.md).

## Set it up

1. **Start the DocRAG backend** (`docker compose up`, or run it as in [GETTING_STARTED.md](GETTING_STARTED.md)). It must answer at http://127.0.0.1:8000, or set `DOCRAG_BACKEND_URL`.
2. **Install the server in its own environment.** Do not use the backend's environment, because the MCP library needs newer versions of `pydantic` and `starlette` than the pinned backend allows.

   ```bash
   python -m venv mcp-env
   mcp-env/Scripts/pip install -r mcp_server/requirements.txt
   ```

3. **Check it starts:** `mcp-env/Scripts/python mcp_server/smoke_test.py` prints `ok:` and the tool names.
4. **Add it to Claude Desktop** (Settings, Developer, Edit config), using the full paths on your machine:

   ```json
   {
     "mcpServers": {
       "docrag": {
         "command": "C:\\path\\to\\mcp-env\\Scripts\\python.exe",
         "args": ["C:\\path\\to\\docrag\\mcp_server\\server.py"],
         "env": { "DOCRAG_BACKEND_URL": "http://127.0.0.1:8000" }
       }
     }
   }
   ```

   Restart Claude Desktop. Then ask something like "Search my documents for the password rules and tell me the page."

Any other MCP client that can start a local program over standard input and output works the same way.

## Notes

- **Errors** are returned as readable messages: the backend is not reachable, a question is empty or too long, or the request timed out.
- **Safety:** the server only talks to the backend address you give it, and the backend listens on this machine only. The tool descriptions mark which tools are read-only. `reindex_documents` is the only one that changes anything, and it only reads the documents folder.
- **Tested:** the tool logic has unit tests, CI starts the server and checks its tool list, and I called every tool through a real MCP client against a running backend, including the language model answer.
- **Not covered:** there is no sign-in, because the server is meant for a local backend. Do not point it at a backend exposed to a network.
