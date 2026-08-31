import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def get_model():
    """Returns a LiteLlm model when a proxy is configured, else a Gemini model name."""
    api_base = os.environ.get("LITELLM_API_BASE")
    if not api_base:
        return os.environ.get("GOOGLE_DEFAULT_MODEL", "gemini-2.5-flash")

    # Imported lazily so the Gemini path does not require google-adk[extensions].
    from google.adk.models.lite_llm import LiteLlm

    # LiteLLM needs an "openai/" routing prefix to treat the proxy as an OpenAI-compatible endpoint
    return LiteLlm(
        model="openai/" + os.environ.get("LITELLM_DEFAULT_MODEL", "gpt-4o-mini"),
        api_base=api_base,
        api_key=os.environ.get("LITELLM_KEY"),
    )
