import os
from dotenv import load_dotenv
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import uvicorn

load_dotenv()

model = OpenAIChatModel(
    os.getenv("LITELLM_DEFAULT_MODEL", "gpt-4o-mini"),
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
)

agent = Agent(model, system_prompt="Provide lead times for material numbers.")


@agent.tool_plain
def get_lead_time(material_id: str) -> str:
    """Returns the production time for a material ID."""
    if material_id.startswith("1"):
        return f"Material {material_id} lead time: 15 business days."
    elif material_id.startswith("2"):
        return f"Material {material_id} lead time: 10 business days."
    else:
        return f"Material {material_id} lead time: 5 business days."


# Convert Pydantic AI agent into an A2A-compatible ASGI app
app = agent_to_a2a(agent)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022)