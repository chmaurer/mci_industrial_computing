import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

@tool
def calculate_tax(subtotal: float, state: str) -> float:
    """Calculates sales tax based on the state code.

    Args:
        subtotal: The monetary amount to apply tax to.
        state: The two-letter state code (e.g., 'CA', 'NY').
    """
    rates = {"NY": 0.08875, "CA": 0.0725, "TX": 0.0625}
    return subtotal * rates.get(state.upper(), 0.0)


# 1. Define your LLM pointing to your LiteLLM proxy
model = ChatOpenAI(
    model=os.environ.get("LITELLM_DEFAULT_MODEL", "gpt-4o-mini"),
    temperature=0,
    base_url=os.environ.get("LITELLM_API_BASE"),
    api_key=os.environ.get("LITELLM_KEY")
)

# 2. Compile them into an automatic agent
# This wraps the model and the tools into a loop that manages the execution for you!
agent = create_agent(model, tools=[calculate_tax])

# 3. Run the agent
user_prompt = "I bought a device for $100 in CA. How much tax do I owe?"
result = agent.invoke({"messages": [("user", user_prompt)]})

# 4. Grab the final assistant message from the conversation history
final_message = result["messages"][-1]
print(final_message.content)