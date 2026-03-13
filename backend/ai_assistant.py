# backend/ai_assistant.py
import os
from groq import Groq, GroqError

# Read API key from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Set it permanently using setx (Windows) or export (Linux/macOS)."
    )

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def ask_ai(prompt: str, cycle_data: dict = None) -> str:
    """
    Ask the AI assistant a question.
    
    Args:
        prompt: The question or instruction for the AI.
        cycle_data: Optional dictionary with user cycle info or context.

    Returns:
        AI-generated text response.
    """
    if cycle_data is None:
        cycle_data = {}  # default empty dict if none provided

    try:
        # Build the conversation messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # If cycle_data exists, add it as another "assistant" or "system" message
        if cycle_data:
            messages.append({"role": "user", "content": f"Context: {cycle_data}"})

        # Call the Groq Python SDK chat completions API
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",  # choose a supported model
        )

        # Return the assistant's text
        return response.choices[0].message.content

    except GroqError as e:
        # Gracefully return an error string instead of crashing
        return f"AI request failed: {str(e)}"