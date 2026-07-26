import asyncio
import os
from uuid import uuid4

import httpx
import requests
from dotenv import load_dotenv


load_dotenv()  # Load environment variables

LITELLM_URL = "http://192.168.1.60:4000"
API_KEY = os.getenv("OPENAI_API_KEY")

# sudo docker run --name=matextract --rm -p 8021:8021 192.168.1.60:8083/repository/dockerrepo/material-extractor:1.4
# sudo docker run --name=matprod --rm -p 8022:8022 192.168.1.60:8083/repository/dockerrepo/material-production:1.1

HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
HEADERS_2 = {"x-litellm-api-key": f"{os.getenv('OPENAI_API_KEY')}"}

def discover_agents():
    """Fetch all configured agents from LiteLLM."""
    response = requests.get(f"{LITELLM_URL}/v1/agents", headers=HEADERS)
    response.raise_for_status()

    # The response is a list of agent objects
    agents_data = response.json()

    # Create a mapping of {name: agent_id}
    # Example: {'agent-material-production': '8ff67c22-...'}
    return {a['agent_name']: a['agent_id'] for a in agents_data}


async def call_a2a(agent_id, text):
    """Call the A2A Gateway."""
    payload = {"message": {"parts": [{"text": text}]}}  # A2A format
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LITELLM_URL}/v1/a2a/{agent_id}/message/send",
            headers=HEADERS,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["message"]["parts"][0]["text"]  # Extract message


async def call_via_litellm(agent_id: str, prompt: str):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    timeout_config = httpx.Timeout(120.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        # 1. SEND (Exactly like before, but Agent ID is handled by LiteLLM)
        send_payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "agent_id": agent_id,
                "message": {
                    "messageId": str(uuid4()),
                    "kind": "message",
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}]
                }
            },
            "id": str(uuid4())
        }

        resp = await client.post(LITELLM_URL+"/v1/a2a/"+agent_id+"/message/send", json=send_payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()

        # Extracting the text from the synchronous result
        # Note: Adjust the keys based on your specific agent's response structure
        result = data.get("result", {})
        history = result.get("parts", [])

        for entry in reversed(history):
            if entry.get("kind") in ["text"]:
                return entry["text"]

        return "No agent response found in history."


async def resolve_agent_by_description(client: httpx.AsyncClient, keyword: str):
    """
    Looks up the /v1/agents list and finds an agent where
    the keyword exists in the description.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}

    # LiteLLM might use 'x-litellm-api-key' or 'Authorization'
    # based on your curl, let's include the specific key header too
    headers["x-litellm-api-key"] = API_KEY

    resp = await client.get(f"{LITELLM_URL}/v1/agents", headers=headers)
    resp.raise_for_status()

    agents = resp.json()
    for agent in agents:
        desc = agent.get("agent_card_params", {}).get("description", "")
        # Check if our keyword is inside the description (case-insensitive)
        if keyword.lower() in desc.lower():
            print(f"Found Agent: {agent.get('agent_name')} ({agent.get('agent_id')})")
            return agent.get("agent_id")

async def run():
    agents = discover_agents()
    print(f"Found agents: {list(agents.keys())}")

    order_text = "We need to schedule production for material '149449' and '255565'."

    # Step 1: Extract numbers
    print("Calling Extraction Agent...")
    # Using exact name from discovery output
    extractor_id = agents.get("agent-material-extraction")
    if not extractor_id:
        print("Error: 'agent-material-extraction' not found in discovered agents.")
        return

    found_numbers = await call_via_litellm(extractor_id, order_text)
    print(f"Numbers found: {found_numbers}")

    # Step 2: Get production time
    print("Calling Production Agent...")
    # Using exact name from discovery output
    production_id = agents.get("agent-material-production")
    if not production_id:
        print("Error: 'agent-material-production' not found in discovered agents.")
        return

    final_info = await call_via_litellm(production_id, f"Lead times for: {found_numbers}")

    print("\n--- FINAL OUTPUT ---")
    print(final_info)


if __name__ == "__main__":
    asyncio.run(run())