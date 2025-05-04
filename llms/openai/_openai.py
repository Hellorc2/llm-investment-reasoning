from openai import OpenAI

from core import settings


openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_llm_response(system_prompt: str, user_prompt: str, temperature: float = 1.0) -> str:
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
        temperature=temperature,
        stream=False
    )
    return response.choices[0].message.content

def get_llm_response_with_history(system_prompt: str, conversation_history: list[dict[str, str]]) -> str:
    """
    Get a response from the OpenAI API, including the full conversation history.
    
    Args:
        system_prompt: The system prompt that sets the context
        conversation_history: List of message dictionaries with 'role' and 'content' keys,
                            representing the back-and-forth conversation between user and assistant
        
    Returns:
        The model's response as a string
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

