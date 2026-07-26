import os
import re
from dotenv import load_dotenv
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import uvicorn
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

model = OpenAIChatModel(
    'gemini-2.5-flash',
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
logger = logging.getLogger(__name__)
agent = Agent(
    model,
    system_prompt=(
        "You are a strict data extraction agent. "
        "CRITICAL RULE: You MUST call the 'extract_material_numbers' tool FIRST on every single query, "
        "even if you think you see numbers in the text. "
        "After receiving the tool result, output ONLY the raw numbers as a comma-separated string (e.g. 149449, 255565). "
        "Do NOT output any other text or explanation."
    )
)


@agent.tool_plain
def extract_material_numbers(text: str) -> list[str]:
    """
    Takes the raw, unedited user string and extracts all material number digit sequences from it,
    excluding any numbers that start with '9'.
    Args:
        text: The raw user prompt in its entirety.
    """
    logger.info(f"Extracting material numbers from text: {text}")
    all_numbers = re.findall(r'\d+', text)

    # Filter out numbers starting with '9'
    filtered_numbers = [num for num in all_numbers if not num.startswith('9')]

    logger.info(f"Filtered Numbers: {filtered_numbers}")
    return filtered_numbers

# Expose as A2A
app = agent_to_a2a(agent)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8021, log_level="info")
