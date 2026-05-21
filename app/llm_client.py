from openai import OpenAI
from app.config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE

client = OpenAI(api_key=OPENAI_API_KEY)


def query_llm(prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Args:
        prompt:      The user-facing prompt string.
        temperature: Sampling temperature (default from TEMPERATURE env var).
                     0 = deterministic, 1 = creative.

    Returns:
        The model's response as a string.

    Raises:
        openai.OpenAIError: On API or network failure (let the caller handle it).
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
