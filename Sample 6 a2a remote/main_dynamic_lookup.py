import asyncio
import os

import httpx
from uuid import uuid4

from dotenv import load_dotenv

from main import resolve_agent_by_description, call_via_litellm

"""
 * Start litellm in version 1.81.14
 * Startup the relevant material-extractor docker image on http://192.168.1.60:8021 (NOT ALWAYS UP!!)
 sudo docker run --name=matextract --rm -p 8021:8021 --net=host 192.168.1.60:8083/repository/dockerrepo/material-extractor:1.4
 * configure the agent and this endpoint in litellm (http://192.168.1.60:8021 is now part of an agent)
 * look up the agents and specifically the "extract material"-agent (see below, resolve_agent_by_description)
 * and request the material extraction (call_via_litellm)
"""
load_dotenv()
# Point to your LiteLLM instance base URL
LITELLM_URL = "http://192.168.1.60:4000"
API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    async with httpx.AsyncClient() as client:
        try:
            # 1. DYNAMIC LOOKUP
            # Search for the agent that handles "extraction"
            target_id = await resolve_agent_by_description(client, "Extracts material")

            # 2. INVOKE
            numbers = await call_via_litellm(target_id, "I want to order 12 pieces of material 252653 and 1 piece of material 252654.")
            print(f"Result: {numbers}")

        except Exception as e:
            print(f"Workflow failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())