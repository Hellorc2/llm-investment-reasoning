import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from llms.openai import get_llm_response


def generate_logical_criteria_doc() -> str:
    """
    Reads if...then... statements from logical_statements.txt and generates an organized document
    outlining logical criteria for startup success.
    
    Returns:
        A structured document organizing the logical criteria
    """
    # Read statements from file
    with open('logical_statements.txt', 'r') as f:
        statements_text = f.read()
    
    system_prompt = """You are a vc analyst tasked with organizing logical criteria 
    for startup success/failure into a clear, structured document, so that one will be able predict the success/failure of a startup."""
    
    user_prompt = f"""Given these if...then... statements about startup success/failure:

{statements_text}

Please organize these statements into a comprehensive document that outlines the logical 
criteria for startup success/failure. Group related criteria together, eliminate redundancies, 
and present the information in a clear, hierarchical structure with sections and subsections.
"""

    return get_llm_response(system_prompt, user_prompt)
