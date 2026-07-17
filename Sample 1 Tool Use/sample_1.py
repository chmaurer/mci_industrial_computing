from langchain_core.tools import tool


@tool
def calculate_tax(subtotal: float, state: str) -> float:
    """Calculates sales tax based on the state code.

    Args:
        subtotal: The monetary amount to apply tax to.
        state: The two-letter state code (e.g., 'CA', 'NY').
    """
    rates = {"NY": 0.08875, "CA": 0.0725, "TX": 0.0625}
    return subtotal * rates.get(state.upper(), 0.0)


# The @tool decorator auto-generates a JSON Schema from the docstring:
print(calculate_tax.args)