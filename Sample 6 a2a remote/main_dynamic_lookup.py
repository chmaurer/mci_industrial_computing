import asyncio
import os
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from main import call_via_litellm

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://192.168.1.62:4000")
API_KEY = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "x-litellm-api-key": API_KEY,
    "accept": "application/json",
}


async def fetch_available_agents(client: httpx.AsyncClient) -> list[dict]:
    """Retrieves all registered agents and their capability metadata."""
    resp = await client.get(f"{LITELLM_URL}/v1/agents", headers=headers)
    resp.raise_for_status()
    return resp.json()


async def route_query_to_agent(client: httpx.AsyncClient, query: str, agents: list[dict]) -> str:
    """Uses a lightweight LLM router to dynamically select the best agent ID based on capabilities."""

    # 1. Format agent metadata into a clean manifest for the router
    agent_manifest = [
        {
            "agent_id": a.get("agent_id"),
            "name": a.get("agent_name"),
            "description": a.get("agent_card_params", {}).get("description", "No description"),
        }
        for a in agents
    ]

    system_prompt = (
        "You are an intent router. Analyze the user request and select the single best "
        "agent ID from the available candidates to process the task. "
        f"Available Agents:\n{agent_manifest}"
    )

    # 2. Ask the router (or LiteLLM proxy) to select the exact agent ID
    # Modern frameworks use Structured Outputs / Function Calling for this
    response = await client.post(
        f"{LITELLM_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": "gpt-4o-mini",  # Use a fast/cheap router model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Select agent for query: {query}"}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_selection",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_agent_id": {"type": "string"}
                        },
                        "required": ["selected_agent_id"]
                    }
                }
            }
        }
    )
    response.raise_for_status()
    selected_id = response.json()["choices"][0]["message"]["content"]

    import json
    return json.loads(selected_id)["selected_agent_id"]

timeout_config = httpx.Timeout(
    connect=10.0,  # 10s connection timeout
    read=60.0,     # 60s read timeout for LLM processing
    write=10.0,    # 10s write timeout
    pool=10.0      # 10s pool wait time
)

async def main():
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            prompt = "I want to order twelve pieces of material 252653 and one piece of material 252654. If 992929 is a material I can order, please ship twelve of it."

            # Fetch available registry
            agents = await fetch_available_agents(client)

            # Let the router pick the right agent based on semantic context
            target_agent_id = await route_query_to_agent(client, prompt, agents)
            print(f"Router assigned query to Agent ID: {target_agent_id}")

            # Execute payload on the target agent
            result = await call_via_litellm(target_agent_id, prompt)
            print(f"Result: {result}")

        except Exception as e:
            print(f"Workflow failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())