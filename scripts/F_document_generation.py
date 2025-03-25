import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from llms.openai import get_llm_response


def generate_logical_criteria_doc(model: str = "openai") -> None:
    """
    Reads if...then... statements from logical_statements.txt and generates an organized document
    outlining logical criteria for startup success. Writes the result to policy.txt.
    
    Args:
        model: The LLM model to use ("openai" or "deepseek")
    """
    # Read statements from file
    with open('logical_statements.txt', 'r') as f:
        statements_text = f.read()
    
    system_prompt = """You are a vc analyst tasked with organizing logical criteria 
    for startup success/failure into a clear, structured document, so that one will be able predict the success/failure of a startup."""
    
    user_prompt = f"""
    Below are multiple sets of logical rules derived from startup reflections:

        {statements_text}

        Combine and consolidate these into one concise, clearly structured final set of logical rules. 
        Remove redundancy and ensure each rule is distinct and actionable. 

        IMPORTANT: You must create EXACTLY:
        - 5 rules for HIGH likelihood of success (numbered 1-5)
        - 5 rules for LOW likelihood of success (numbered 6-10)

        FORMATTING REQUIREMENTS:
        1. Start with "HIGH LIKELIHOOD OF SUCCESS RULES:" header
        2. List exactly 5 HIGH success rules, numbered 1-5
        3. Then add "LOW LIKELIHOOD OF SUCCESS RULES:" header
        4. List exactly 5 LOW success rules, numbered 6-10
        5. Each rule must be on its own line
        6. Each rule must start with its number followed by a period
        
        Example format:
        HIGH LIKELIHOOD OF SUCCESS RULES:
        1. Category: Technical Background
           IF [founder has PhD in relevant field] AND [has 5+ years industry experience] THEN likelihood_of_success = HIGH
        2. Category: Another High Success Rule...
        (and so on until rule 5)
        
        LOW LIKELIHOOD OF SUCCESS RULES:
        6. Category: Market Timing
           IF [market is highly saturated] AND [no clear differentiation] THEN likelihood_of_success = LOW
        7. Category: Another Low Success Rule...
        (and so on until rule 10)

        REMEMBER: You MUST output exactly 5 rules for each category, numbered 1-5 for HIGH and 6-10 for LOW.
"""

    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    result = get_llm_response(system_prompt, user_prompt)
    print(result)
    
    # Write result to policy.txt
    with open('policy.txt', 'w') as f:
        f.write(result)
