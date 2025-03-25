import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import settings
from datetime import datetime
from data_utils import founder_df, get_n_filtered_rows
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
             Founder Profile: {founder_profile}
        This startup was eventually {success_text}.
        Clearly explain the most important reasons why this startup {success_verb}."""
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    return get_llm_response(system_prompt, user_prompt)
    


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









