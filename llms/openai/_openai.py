from openai import OpenAI

from core import settings


openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    """
    Get a response from the OpenAI API.
    
    Args:
        system_prompt: The system prompt that sets the context
        user_prompt: The user's specific question or request
        
    Returns:
        The model's response as a string
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

