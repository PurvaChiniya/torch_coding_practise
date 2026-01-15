# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoServer")  # name shows up in clients

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    print(f"[server] add({a}, {b})")
    return a + b

@mcp.resource("file://./notes.txt")
def get_notes() -> str:
    """Return the content of my_notes.txt."""
    try:
        with open("my_notes.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Note file not found."

if __name__ == "__main__":
    mcp.run()  # stdio transport; perfect for local dev

