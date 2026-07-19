import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, AzureChatOpenAI

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


# Initialize the model and bind the tool schema directly to it
# Initialize the model pointing to your LiteLLM proxy
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    base_url=os.environ.get("LITELLM_API_BASE"),
    api_key=os.environ.get("LITELLM_KEY"),
   # api_version="2025-01-01-preview"
)
model_with_tools = model.bind_tools([calculate_tax])

# 1. Ask the question requiring a tool
user_prompt = "I bought a device for $100 in CA. How much tax do I owe?"
response = model_with_tools.invoke(user_prompt)

# Inspect the structured request from the LLM
print("LLM Tool Calls:", response.tool_calls)

# 2. Extract and actually RUN the tool
tool_call = response.tool_calls[0]
tool_result = calculate_tax.invoke(tool_call["args"])  # Run the Python function
print("Tool Result:", tool_result)

# 3. Feed the entire history back to the model:
#    - The original question
#    - The LLM's decision to call the tool
#    - The result of the tool execution
final_response = model_with_tools.invoke([
    HumanMessage(content=user_prompt),
    response,  # This contains the tool call request
    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])  # The tool output
])

print("\nFinal LLM Response:")
print(final_response.content)