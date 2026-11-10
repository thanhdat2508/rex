"""MCP (Model Context Protocol) stdio server for Rex (optional extra).

Install the extra to use it:

    pip install "rex[mcp]"

Then run the server:

    rex-mcp-server            # console script
    python -m rex.mcp.server  # equivalent module form

The submodules ``device`` and ``tools`` are importable without the ``mcp``
dependency; ``server`` requires the extra.
"""

__all__ = ["device", "server", "tools"]
