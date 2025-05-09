from openai import OpenAI

from core import settings


deepseek_client = OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def get_llm_response(system_prompt: str, user_prompt: str, model: str = "deepseek-chat") -> str:
    """
    Get a response from the OpenAI API.
    
    Args:
        system_prompt: The system prompt that sets the context
        user_prompt: The user's specific question or request
        
    Returns:
        The model's response as a string
    """
    response = deepseek_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def get_llm_response_with_history(system_prompt: str, conversation_history: list[dict[str, str]], model: str = "deepseek-chat") -> str:
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
    
    response = deepseek_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

