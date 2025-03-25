import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from llms.openai import get_llm_response
from typing import List, Dict
from datetime import datetime

def analyze_insights(csv_path: str = 'founder_insights.csv', model: str = "openai") -> List[Dict]:
    """
    Read insights from CSV and analyze patterns using LLM.
    
    Args:
        csv_path: Path to the CSV file containing founder insights
        model: The LLM model to use ("openai" or "deepseek")
        
    Returns:
        List of dictionaries containing analysis results
    """
    # Read the insights CSV
    df = pd.read_csv(csv_path)

    logical_statements = []
    
    # Analyze patterns across insights
    system_prompt = """You are a mathematician / logician trying to understand the patterns of success and 
    failure of startups, by converting a description of the founder's profile and the startup's journey into
      a list of logical statements."
    """
    for _, row in df.iterrows():
        founder = row['founder']
        insight = row['insight']
        success = row['success']
        
        user_prompt = f"""Given this insight about a founder whose startup is {success}:

        {insight}

        Based on these reflections, clearly derive a concise set of logical rules in the structured format below:

        IF [condition A] AND [condition B] AND [condition C] THEN likelihood_of_success = HIGH (or LOW)

        Limit the number of rules to 3–5 concise, actionable, and generalizable conditions.
        
        {insight}"""
        

        if model == "openai":
            from llms.openai import get_llm_response
        else:
            from llms.deepseek import get_llm_response
            
        logical_statements.append(get_llm_response(system_prompt, user_prompt))

    return logical_statements