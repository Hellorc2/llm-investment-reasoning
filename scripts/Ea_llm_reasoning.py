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

        IF [condition A1] AND [condition A2] AND ... AND [condition An] THEN likelihood_of_success = [number between 0 and 1]

        Note that n is the number of conditions and can be any number. 
        
        Finally, limit the number of rules to 3–5 concise, actionable, and generalizable conditions.
        
        """
        

        if model == "openai":
            from llms.openai import get_llm_response
        else:
            from llms.deepseek import get_llm_response
            
        logical_statements.append(get_llm_response(system_prompt, user_prompt))

    return logical_statements

def analyze_insights_in_groups(csv_path: str = 'founder_insights.csv', model: str = "openai") -> str:
    """
    Read insights from CSV and analyze patterns across all insights together using LLM.
    
    Args:
        csv_path: Path to the CSV file containing founder insights
        model: The LLM model to use ("openai" or "deepseek")
        
    Returns:
        A comprehensive analysis of all insights combined
    """
    # Read the insights CSV
    df = pd.read_csv(csv_path)
    
    # Combine all insights into a single string
    combined_insights = []
    for _, row in df.iterrows():
        founder = row['founder']
        insight = row['insight']
        success = "successful" if row['success'] else "unsuccessful"
        combined_insights.append(f"Founder: {founder}\nOutcome: {success}\nInsight: {insight}\n---")
    
    combined_text = "\n\n".join(combined_insights)
    
    # Analyze patterns across all insights
    system_prompt = """You are a mathematician / logician trying to understand the patterns of success and 
    failure of startups, by converting a description of the founder's profile and the startup's journey into
      a list of logical statements."
    """
    
    user_prompt = f"""Given these insights about multiple founders and their startup outcomes:

    {combined_text}

    Based on these reflections, clearly derive a concise set of logical rules in the structured format below:

        IF [condition A1] AND [condition A2] AND ... AND [condition An] THEN likelihood_of_success = [number between 0 and 1]

        Note that n is the number of conditions and can be any number, but try to have around 2 ANDs in each rule.

        Also, if you think a rule is neutral (i.e. moderate probability), then don't include it.

        For the conditions, you are only allowed to use one of the following:
        professional_athlete	childhood_entrepreneurship	competitions	ten_thousand_hours_of_mastery	
        languages	perseverance	risk_tolerance	vision	adaptability	personal_branding	
        education_level	education_institution	education_field_of_study	education_international_experience	
        education_extracurricular_involvement	education_awards_and_honors	big_leadership	nasdaq_leadership	
        number_of_leadership_roles	being_lead_of_nonprofits	number_of_roles	number_of_companies	industry_achievements	
        big_company_experience	nasdaq_company_experience	big_tech_experience	google_experience	facebook_meta_experience	
        microsoft_experience	amazon_experience	apple_experience	career_growth	moving_around	
        international_work_experience	worked_at_military	big_tech_position	worked_at_consultancy	worked_at_bank	
        press_media_coverage_count	vc_experience	angel_experience	quant_experience	
        board_advisor_roles	tier_1_vc_experience	startup_experience	ceo_experience	investor_quality_prior_startup	
        previous_startup_funding_experience
        ipo_experience, num_acquisitions, domain_expertise, skill_relevance, yoe   
        .
        
        Finally,limit the number of rules to around 50 actionable and generalizable conditions. Make sure at least half of 
        the rules focus on when the founder is more likely to fail. 
    """
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    response = get_llm_response(system_prompt, user_prompt)
    # Write the logical rules to a file
    with open('logical_statements.txt', 'w') as f:
        f.write(response)
    
    return response

def logical_statements_to_csv(txt_path: str = 'logical_statements.txt', model: str = "openai") -> str:
    """
    Read insights from CSV and analyze patterns across all insights together using LLM.
    
    Args:
        csv_path: Path to the CSV file containing founder insights
        model: The LLM model to use ("openai" or "deepseek")
        
    Returns:
        A comprehensive analysis of all insights combined
    """
    # Read the insights CSV
    # Read the logical statements from file

    with open(txt_path, 'r') as f:
        combined_text = f.read()
    # Analyze patterns across all insights
    
    system_prompt = """You are a mathematician / logician trying to understand the patterns of success and 
    failure of startups, by converting a description of the founder's profile and the startup's journey into
      a list of logical statements."
    """
    
    user_prompt = f"""Given these set of rules:

    {combined_text}

    Convert each rule into a csv row with the format successpredictor(Boolean variable 0 or 1), condition1, condition2, ..., conditionn, likelihood_of_success

    For the conditions, you are still only allowed to use one of the following:
    professional_athlete	childhood_entrepreneurship	competitions	ten_thousand_hours_of_mastery	
    languages	perseverance	risk_tolerance	vision	adaptability	personal_branding	
    education_level	education_institution	education_field_of_study	education_international_experience	
    education_extracurricular_involvement	education_awards_and_honors	big_leadership	nasdaq_leadership	
    number_of_leadership_roles	being_lead_of_nonprofits	number_of_roles	number_of_companies	industry_achievements	
    big_company_experience	nasdaq_company_experience	big_tech_experience	google_experience	facebook_meta_experience	
    microsoft_experience	amazon_experience	apple_experience	career_growth	moving_around	
    international_work_experience	worked_at_military	big_tech_position	worked_at_consultancy	worked_at_bank	
    press_media_coverage_count	vc_experience	angel_experience	quant_experience	
    board_advisor_roles	tier_1_vc_experience	startup_experience	ceo_experience	investor_quality_prior_startup	
    previous_startup_funding_experience
    ipo_experience, num_acquisitions, domain_expertise, skill_relevance, yoe   

    return me only the csv rows, no other text.

    """
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    response = get_llm_response(system_prompt, user_prompt)
    # Write the logical rules to a file
    with open('logical_statements_preprocessed.txt', 'w') as f:
        f.write(response)
    
    return response