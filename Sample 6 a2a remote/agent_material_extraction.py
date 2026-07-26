import os
import re
from dotenv import load_dotenv
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import uvicorn

# 1. Load .env at the beginning
load_dotenv()

# 2. Configure model to point to your LiteLLM instance
# Uses OPENAI_BASE_URL and OPENAI_API_KEY from your .env
model = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

agent = Agent(model, system_prompt="You are an extractor. Find material number sequences.")

@agent.tool_plain
def extract_material_numbers(text: str) -> list[str]:
    """Extracts all digit sequences from the provided text."""
    return re.findall(r'\d+', text)

# Expose as A2A
app = agent_to_a2a(agent)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021)