import asyncio
import os

import httpx
from uuid import uuid4

from dotenv import load_dotenv

from main import resolve_agent_by_description, call_via_litellm

"""
 This file shows that you do not have to hardcode the agent IDs, but you can look them up based on a textual 
 description of what you want to achieve.
"""
load_dotenv()
# Point to your LiteLLM instance base URL
LITELLM_URL = "http://192.168.1.62:4000"
API_KEY = os.getenv("OPENAI_API_KEY")


async def main():
    async with httpx.AsyncClient() as client:
        try:
            # 1. DYNAMIC LOOKUP
            # Search for the agent that handles "extraction"
            target_id = await resolve_agent_by_description(client, "Extracts material")

            # 2. INVOKE
            numbers = await call_via_litellm(target_id,
                                             "I want to order twelve pieces of material 252653 and one piece of material 252654.")
            print(f"Result: {numbers}")

        except Exception as e:
            print(f"Workflow failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
