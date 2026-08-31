import asyncio
import json
import os

from dotenv import load_dotenv
from mcp.client.streamable_http import streamable_http_client
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# If your server uses the newer Streamable HTTP spec instead of SSE, uncomment below:
# from mcp.client.streamable_http import streamable_http_client

load_dotenv()

# 1. Initialize OpenAI Client pointing to LiteLLM Proxy Gateway
llm_client = OpenAI(
    base_url=os.getenv("LITELLM_API"),
    api_key=os.getenv("LITELLM_KEY")
)

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
MODEL_NAME = os.getenv("LITELLM_DEFAULT_MODEL", "gpt-4o-mini")


def convert_mcp_to_openai_tools(mcp_tools):
    """Converts tools discovered from MCP server into OpenAI JSON schema format."""
    openai_tools = []
    for tool in mcp_tools.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}}
            }
        })
    return openai_tools


async def run_react_agent(user_prompt: str):
    print(f"Connecting to MCP Server at: {MCP_SERVER_URL}...")

    # 2. Open an SSE transport channel to your containerized MCP server
    # (Note: Use streamable_http_client if server uses HTTP transport)
    async with streamable_http_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("Connected!")

            # 3. Dynamically discover tools from the MCP Server
            mcp_tools_list = await session.list_tools()
            openai_tools = convert_mcp_to_openai_tools(mcp_tools_list)

            print(f"Discovered Tools: {[t['function']['name'] for t in openai_tools]}\n")

            # Initialize conversation history with system instructions
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful ReAct assistant. Use available tools when required to answer calculations accurately."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            # 4. START REACT AGENT LOOP
            while True:
                print("🧠 [Reasoning] Querying LLM via LiteLLM...")

                response = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto"
                )

                response_message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Case A: LLM produced a final textual response (ReAct Loop Complete)
                if not response_message.tool_calls:
                    print("\n--- Final Synthesis Result ---")
                    print(response_message.content)
                    break

                # Case B: LLM wants to execute one or more tools (Acting)
                messages.append(response_message)  # Append assistant's call to history

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    print(f"⚡ [Action] Executing MCP Tool '{function_name}' with arguments: {arguments}")

                    # Execute tool remotely via MCP Client Session
                    try:
                        result = await session.call_tool(function_name, arguments=arguments)

                        # Extract string representation from tool output content
                        tool_output = ""
                        if result.content:
                            tool_output = result.content[0].text
                        else:
                            tool_output = str(result.structured_content)

                        print(f"📥 [Observation] Result from MCP Server: {tool_output}\n")

                    except Exception as e:
                        tool_output = f"Error executing tool: {str(e)}"
                        print(f"❌ [Observation] Tool Failed: {tool_output}\n")

                    # Feed the observation back into the conversation context
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(tool_output)
                    })


if __name__ == "__main__":
    prompt = "Calculate the state sales tax for a $100 purchase in California."
    asyncio.run(run_react_agent(prompt))