import asyncio
import os
from uuid import uuid4
import httpx
import requests
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

LITELLM_URL = "http://192.168.1.62:4000"
API_KEY = os.getenv("OPENAI_API_KEY")

# register agents on litellm
# add IPs to user_url_allowed_hosts in litellm config.yaml
# sudo docker run --name=matextract --rm -p 8021:8021 192.168.1.62:8083/repository/dockerrepo/material-extractor:1.4
# sudo docker run --name=matprod --rm -p 8022:8022 192.168.1.62:8083/repository/dockerrepo/material-production:1.1

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
            f"{LITELLM_URL}/v1/a2a/{agent_id}/message/send",
            json=send_payload,
            headers=headers
        )
        resp.raise_for_status()

        data = resp.json()

        # Flexibly parse A2A / JSON-RPC structures
        parts = data.get("result", {}).get("message", {}).get("parts", [])
        if not parts:
            parts = data.get("result", {}).get("parts", [])

        for entry in reversed(parts):
            # Check text in various possible positions
            if isinstance(entry, dict):
                if entry.get("kind") == "text" and "text" in entry:
                    return entry["text"].strip()
                if "content" in entry:
                    return str(entry["content"]).strip()
                if "text" in entry:
                    return str(entry["text"]).strip()

        # Fallback to stringifying the raw result if structure is unknown
        if "result" in data:
            return str(data["result"]).strip()

        return "No agent response found in history."


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
