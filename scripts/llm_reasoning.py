import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from llms.openai import get_llm_response
from typing import List, Dict
from datetime import datetime

def analyze_insights_in_groups(csv_path: str = 'founder_insights.csv', model: str = "openai", iterative: bool = False, previous_analysis: str = "") -> str:
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

        IF [condition A1] AND [condition A2] AND ... AND [condition An] THEN probability_of_success/failure = [number between 0 and 1]

        Note that n is the number of conditions and can be any number. Try to have around 2 ANDs in each rule, but play around with the number of ANDs to 
        get a variety of rules. AVOID using ORs!

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
        ipo_experience, num_acquisitions, domain_expertise, skill_relevance
        .

        If it's a negative condition (e.g. low, none, false, etc.), add a "not_" in front of the condition.
        
        Finally,limit the number of rules to between 18 and 21 actionable and generalizable conditions. Make sure at least 12 of 
        the rules focus on when the founder is more likely to succeed, and at least 6 of the rules focus on when the founder is more likely to fail. If 
        two rules are very similar, try to merge them into one rule or find a different rule to replace one of them.

    """

    if iterative:
        user_prompt += f"""You are also given your analysis on a previous set of founders, where the probabilities were adjusted based on real world data:

        {previous_analysis}.

        Consider these old rules and the probabilities as hints, but you are STRONGLY encouraged to add new rules, delete existing rules
        or slightly modify the probabilities based on your new thoughts. Bring the total number of rules to between 18 and 21. Again, you are
        encouraged to experiment with the rules and don't be afraid to go against the hints."""
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    response = get_llm_response(system_prompt, user_prompt, model = "deepseek-chat")
    # Write the logical rules to a file
    with open('logical_statements/logical_statements.txt', 'w') as f:
        f.write(response)
    
    return response

def logical_statements_preprocess(input_file: str = 'logical_statements/logical_statements.txt', output_file: str = 'logical_statements/logical_statements_preprocessed.txt', 
                                  model: str = "openai", iterative_index: int = 0) -> str:
    """
    Read logical statements from a text file and preprocess them using LLM.
    
    Args:
        txt_path (str): Path to the input text file containing logical statements
        model (str): The LLM model to use ("openai" or "deepseek")
        
    Returns:
        str: The preprocessed logical statements
    """
    # Read the logical statements from file
    with open(input_file, 'r') as f:
        combined_text = f.read()
    # Analyze patterns across all insights
    
    system_prompt = """You are a mathematician / logician trying to understand the patterns of success and 
    failure of startups, by converting a description of the founder's profile and the startup's journey into
      a list of logical statements."
    """
    
    user_prompt = f"""Given these set of rules (if there are other text around, look for something like "proposed policy", or figure out yourself which
    text is the set of rules):

    {combined_text}

    Firstly, filter out any rules that contains ORs (completely).
    
    Then convert each rule into a csv row with the format successpredictor(Boolean variable 0 or 1),condition1,condition2,...,conditionn,likelihood_of_success

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
    ipo_experience, num_acquisitions, domain_expertise, skill_relevance  

    DO NOT include comparatives in the conditions such as education_level = 'high'. If you see a condition like this with positive characteristics 
    such as high, >, true, remove everything including and after the comparator; if you see a condition like this with negative characteristics such as 
    low, <, false, do the same but add not_ in front of the condition.

    DOUBLE CHECK that all the conditions appear in the list of allowed conditions, or is one of the conditions with "not_" in front, after you take into consideration
    the above instructions on interpreting comparatives! If not, delete the rule.

    Return me ONLY the csv rows, no other text.
    """
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    response = get_llm_response(system_prompt, user_prompt)
    # Write the logical rules to a file

    
    with open(output_file, 'w') as f:
        f.write(response)
    
    return response

def modify_analysis_based_on_advice(analysis_path: str = 'iterative_training_results/iteration_000_preprocessed.txt', iterative_index = 0, model: str = "deepseek"):
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'r') as f:
        evaluated_str = """"Here are some advices on how to modify the policy to make work better. Modify these rules based on the advices:""" + f.read()
    
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'r') as f:
        combined_text = f.read()

        system_prompt = """You are a VC analyst trying to produce a policy for predicting startup success. You have produced previous policies and are now
        given some advices on how to modify the policy to make it work better. for the policy, each rule has the format 
        successpredictor(Boolean variable 0 or 1), condition1, condition2, ..., conditionk, probability of success/failure, where the conditions are ANDed together."
    """
    
    user_prompt = f"""Given the policy produced at the current iteration:

    {combined_text}

    And the advices on how to modify the policy:

    {evaluated_str}

    Modify the policy based on the advices, using the same format.

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
    ipo_experience, num_acquisitions, domain_expertise, skill_relevance  

    DO NOT include comparatives in the conditions such as education_level = 'high'. If you see a condition like this with positive characteristics 
    such as high, >, true, remove everything including and after the comparator; if you see a condition like this with negative characteristics such as 
    low, <, false, do the same but add not_ in front of the condition.

    DOUBLE CHECK that all the conditions appear in the list of allowed conditions, or is one of the conditions with "not_" in front! If not, 
    delete the rule.
    
    return me only the csv rows, no other text.

    """
    
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    response = get_llm_response(system_prompt, user_prompt, model = "deepseek-chat")

    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'w') as f:
        f.write(response)

    return response