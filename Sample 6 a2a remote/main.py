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

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
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

        resp = await client.post(
            LITELLM_URL + "/v1/a2a/" + agent_id + "/message/send",
            json=send_payload,
            headers=headers
        )
        resp.raise_for_status()

        data = resp.json()
        result = data.get("result", {})
        history = result.get("parts", [])

        # Parse text or raw content from the response parts
        for entry in reversed(history):
            if entry.get("kind") == "text":
                return entry.get("text", "").strip()
            elif entry.get("kind") == "data":
                return str(entry.get("data", "")).strip()

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

    order_text = (""
                  "Hi there! "
                  ""
                  "I have heard that you do produce products I am really fond of. I have looked up your products on "
                  "the web and I'd like to get an order started for items 149449 and 255565. However, the order numbers could also be 90393032 or 9382388, I was unsure to identify them correctly on the website."
                  ""
                  "Could you let me know what the production timeline looks like for both?")

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
