# mcp_combined.py
import asyncio
import sys
from fastmcp import FastMCP

# =====================================================================
# SYSTEM 1: THE SERVER DEFINITION
# =====================================================================
server_mcp = FastMCP("MCI-Industrial-Computing")


# 1. DEFINE A TOOL (Executable function with type hints)
@server_mcp.tool()
def compute_square(number: int) -> int:
    """Calculates the square of a given integer number."""
    return number * number


# 2. DEFINE A RESOURCE (Read-only data layer)
@server_mcp.resource("archive://lecture-notes/mcp-intro")
def read_lecture_notes() -> str:
    """Provides the read-only introductory text regarding MCP protocols."""
    return "Model Context Protocol (MCP) successfully decouples LLM applications from tools!"


# 3. DEFINE A PROMPT (Workflow templates)
@server_mcp.prompt()
def explain_code_style(language: str) -> str:
    """A preset template instructing an LLM on how to explain code architecture."""
    return f"Act as a senior software architect. Explain the best practices for writing clean code in {language}."


# =====================================================================
# SYSTEM 2: THE WRAPPER & CLIENT RUNTIME
# =====================================================================
async def run_client_orchestrator():
    # Wait half a second to ensure the channel opens smoothly
    await asyncio.sleep(0.5)

    print("\n=== 🔌 WRAPPER: INITIALIZING BACKGROUND SERVER VIA STDIO ===")

    # We spawn a background process using 'uv run python mcp_combined.py --server-mode'
    # This acts as our distinct, decoupled server infrastructure
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp import ClientSession

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", __file__, "--server-mode"]
    )

    # Establish connection channel
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Handshake exchange
            await session.initialize()
            print("✔ Handshake Successful. Client session initialized natively.\n")

            # 1. Discover Tools
            print("--- 🛠 DISCOVERING MCP TOOLS ---")
            tools_response = await session.list_tools()
            for t in tools_response.tools:
                print(f"-> Found Tool: [ {t.name} ]")
                print(f"   Schema: {t.inputSchema}\n")

            # 2. Discover & Read Resources
            print("--- 📚 DISCOVERING MCP RESOURCES ---")
            resources_response = await session.list_resources()
            for r in resources_response.resources:
                print(f"-> Found URI: {r.uri}")
                content = await session.read_resource(r.uri)
                print(f"   Server Live Payload: \"{content.contents[0].text}\"\n")

            # 3. Discover & Fetch Prompts
            print("--- 🎯 DISCOVERING MCP PROMPTS ---")
            prompts_response = await session.list_prompts()
            for p in prompts_response.prompts:
                print(f"-> Found Prompt Template: [ {p.name} ]")
                print(f"   Description: {p.description}")
                # Print the arguments that the template requires (e.g., 'language')
                print(f"   Required Arguments: {[arg.name for arg in p.arguments] if p.arguments else []}")

                # Let's actively request the server to compile the prompt for "Python"
                prompt_get_result = await session.get_prompt(p.name, arguments={"language": "Python"})
                # Extra detail: Extract the actual string text out of the compiled prompt messages payload
                compiled_text = prompt_get_result.messages[0].content.text
                print(f"   Compiled Server Output: \"{compiled_text}\"\n")

            # 4. Request Execution Layer
            print("--- ⚡ TRIGGERING DECOUPLED TOOL EXECUTION ---")
            execution_result = await session.call_tool("compute_square", arguments={"number": 15})
            print(f"Result returned from isolated server: {execution_result.content[0].text}\n")


# =====================================================================
# SYSTEM 3: ENTRYPOINT ROUTER
# =====================================================================
if __name__ == "__main__":
    # If the process is spawned with the flag, it behaves purely as a server
    if "--server-mode" in sys.argv:
        server_mcp.run()
    else:
        # Otherwise, it runs the client application wrapper
        asyncio.run(run_client_orchestrator())