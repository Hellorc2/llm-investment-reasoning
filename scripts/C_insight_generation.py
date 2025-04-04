import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import settings
from Z_data_utils import get_n_filtered_rows
from datetime import datetime
from typing import List, Tuple

def analyze_founder(founder_profile: str, success: bool, model: str = "openai") -> str:
    """
    Analyze a founder's profile and explain why their startup succeeded or failed.
    
    Args:
        founder_profile: The founder's background and startup idea
        success: Whether the startup was successful or not
        model: The LLM model to use ("openai" or "deepseek")
        
    Returns:
        A detailed explanation of why the startup succeeded or failed
    """
    success_text = "successful" if success else "unsuccessful"
    success_verb = "succeeded" if success else "failed"
    
    system_prompt = "You are a VC anaylst trying to predict the success of a startup based some information about the founders."
    
    user_prompt = f"""Given the following founder's background and startup idea:
             Founder Profile (LinkedIn followed by Crunchbase): {founder_profile}
             Predict whether or not this startup was successful. Clearly explain your arguments. 
             Try to use simple vocabulary"""
    
    if model == "openai":
        from llms.openai import get_llm_response, get_llm_response_with_history
    else:
        from llms.deepseek import get_llm_response, get_llm_response_with_history
        
    prediction = get_llm_response(system_prompt, user_prompt)
    print(prediction)

    user_prompt2 = f"""You are now given that the same founder was {success_text}. Please reflect on your prediction. If your prediction was correct,
    try to polish your arguments. If your prediction was incorrect, think about what might went wrong and produce a new explanation. In either
    case, clearly identify the most important reasons why the startup was {success_verb}."""

    reflection = get_llm_response_with_history(system_prompt, [{"role": "user", "content": user_prompt}, 
                                                               {"role": "assistant", "content": prediction}, 
                                                               {"role": "user", "content": user_prompt2}])
    print(reflection)


    return reflection


    """
    user_prompt2 = Can you rephrase each sentence in the above insight in a more logical way? 
    DON'T drop or add any sentences. 

    conversation_history = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": insight},
        {"role": "user", "content": user_prompt2}
    ]
    
    return get_llm_response_with_history(system_prompt, conversation_history)

    """



def save_result(timestamp: str, result: str) -> None:
    """
    Save the analysis result to a log file.
    
    Args:
        timestamp: The timestamp for the session
        result: The analysis result to save
    """
    # Create records directory if it doesn't exist
    os.makedirs('records', exist_ok=True)

    # Create the log file with timestamp in the records directory
    log_filename = os.path.join('records', f'log_{timestamp}.txt')
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"Session: {timestamp}\n")
        f.write(f"{'='*50}\n")
        f.write(result)
        f.write("\n")


"""
        Generate insights about a founder's success or failure using OpenAI's API.

        status = "successful" if startup_success else "unsuccessful"
        outcome = "succeeded" if startup_success else "failed"

        # Format the lists into readable strings
        degrees_str = "\n- " + "\n- ".join(university_degrees) if university_degrees else "No university degrees listed"
        work_str = "\n- " + "\n- ".join(work_history) if work_history else "No work history listed"
        companies_str = "\n- " + "\n- ".join(previous_companies_founded) if previous_companies_founded else "No previous companies founded"

        prompt = f
        Founder Name: {founder_name}
        Location: {company_location}
        
        Education:
        {degrees_str}
        
        Work History:
        {work_str}
        
        Previous Companies Founded:
        {companies_str}
        
        Professional Background and Achievements:
        {professional_background}
        
        This startup was eventually {status}.
"""








