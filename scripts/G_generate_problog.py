import pandas as pd

def read_preprocessed_rules(iteration_number):
    """
    Read preprocessed rules from the iteration_{number}_preprocessed.txt file
    
    Args:
        iteration_number (str): Iteration number to read from
        
    Returns:
        list: List of tuples containing (is_success, feature1, feature2, probability)
    """
    rules = []
    filename = f"iterative_training_results/iteration_{iteration_number:03d}_preprocessed.txt"
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if line.startswith('1') or line.startswith('0'):
                # Parse the line into components
                is_success = line.startswith('1')
                
                # Extract features and probability
                parts = line.split(',')
                # Get probability from last element
                probability = parts[-1].strip()                
                # Join all other parts except the first (is_success indicator) for feature string
                feature_str = ','.join(parts[1:-1])
                rules.append((is_success, feature_str, probability))
    return rules

def read_polished_rules(iteration_number):
    """
    Read polished rules from the iteration_{number}_polished.txt file
    
    Args:
        iteration_number (str): Iteration number to read from
        
    Returns:
        list: List of tuples containing (is_success, feature1, feature2, probability)
    """
    rules = []
    filename = f"iterative_training_results/iteration_{iteration_number:03d}_polished.txt"
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if line.startswith('Success rule:') or line.startswith('Failure rule:'):
                # Parse the line into components
                is_success = line.startswith('Success rule:')
                
                # Extract features and probability
                parts = line.split(', probability: ')
                feature_str = parts[0].split(': ')[1]
                probability = parts[1].strip()
                rules.append((is_success, feature_str, probability))
    return rules

def generate_base_problog_program(iteration_number):
    attributes = [
        "professional_athlete", "childhood_entrepreneurship", "competitions", "ten_thousand_hours_of_mastery",
        "languages", "perseverance", "risk_tolerance", "vision", "adaptability", "personal_branding",
        "education_level", "education_institution", "education_field_of_study", "education_international_experience",
        "education_extracurricular_involvement", "education_awards_and_honors", "big_leadership", "nasdaq_leadership",
        "number_of_leadership_roles", "being_lead_of_nonprofits", "number_of_roles", "number_of_companies", "industry_achievements",
        "big_company_experience", "nasdaq_company_experience", "big_tech_experience", "google_experience", "facebook_meta_experience",
        "microsoft_experience", "amazon_experience", "apple_experience", "career_growth", "moving_around",
        "international_work_experience", "worked_at_military", "big_tech_position", "worked_at_consultancy", "worked_at_bank",
        "press_media_coverage_count", "vc_experience", "angel_experience", "quant_experience",
        "board_advisor_roles", "tier_1_vc_experience", "startup_experience", "ceo_experience", "investor_quality_prior_startup",
        "previous_startup_funding_experience", "ipo_experience", "num_acquisitions", "domain_expertise", "skill_relevance", "yoe"
    ]

    rules = read_preprocessed_rules(iteration_number)
    with open(f'problog_program_base_{iteration_number}.pl', 'w') as f:
        for rule in rules:
            is_success, feature_str, probability = rule
            # Clean up feature string by removing brackets and quotes
            features = feature_str.strip('[]').replace("'", "").split(', ')
            # Format the problog line
            if is_success:
                pred = "success"
            else:
                pred = "failure"
            
            conditions = ','.join(features)

            # Replace not_ with ~ in conditions
            conditions = conditions.replace('not_', '\+')
            
            if probability == "not enough samples":
                probability = "0.5" # Default probability for unknown cases
                
            problog_line = f"{probability}::{pred} :- {conditions}.\n"
            f.write(problog_line)
        
        # Add query for success probability
        f.write("\nquery(success).\n")
        f.write("\nquery(failure).\n")




def generate_problog_program(iteration_number, founder_info, program_file):

    # Define all possible attributes
    attributes = [
        "professional_athlete", "childhood_entrepreneurship", "competitions", "ten_thousand_hours_of_mastery",
        "languages", "perseverance", "risk_tolerance", "vision", "adaptability", "personal_branding",
        "education_level", "education_institution", "education_field_of_study", "education_international_experience",
        "education_extracurricular_involvement", "education_awards_and_honors", "big_leadership", "nasdaq_leadership",
        "number_of_leadership_roles", "being_lead_of_nonprofits", "number_of_roles", "number_of_companies", "industry_achievements",
        "big_company_experience", "nasdaq_company_experience", "big_tech_experience", "google_experience", "facebook_meta_experience",
        "microsoft_experience", "amazon_experience", "apple_experience", "career_growth", "moving_around",
        "international_work_experience", "worked_at_military", "big_tech_position", "worked_at_consultancy", "worked_at_bank",
        "press_media_coverage_count", "vc_experience", "angel_experience", "quant_experience",
        "board_advisor_roles", "tier_1_vc_experience", "startup_experience", "ceo_experience", "investor_quality_prior_startup",
        "previous_startup_funding_experience", "ipo_experience", "num_acquisitions", "domain_expertise", "skill_relevance", "yoe"
    ]

    
    attribute_limits = {
        "languages": [1, 4],
        "perseverance": [0, 2],
        "risk_tolerance": [0, 2],
        "vision": [0, 2],
        "adaptability": [0, 2],
        "personal_branding": [0, 2],
        "education_level": [0, 3],
        "education_institution": [0, 4],
        "education_field_of_study": [0, 3],
        "big_leadership": [0, 3],
        "nasdaq_leadership": [0, 3],
        "number_of_leadership_roles": [0, 2],
        "number_of_roles": [0, 20],
        "number_of_companies": [0, 10],
        "industry_achievements": [0, 1],
        "press_media_coverage_count": [0, 2],
        "vc_experience": [0, 2],
        "angel_experience": [0, 2],
        "quant_experience": [0, 2],
        "investor_quality_prior_startup": [0, 2],
        "previous_startup_funding_experience": [0, 4],
        "num_acquisitions": [0, 1],
        "domain_expertise": [0, 3],
        "skill_relevance": [0, 3],
    }



    with open(program_file, 'w') as f:
        for attr in attributes:
            # Get the value from founder_info dataframe row
            value = founder_info[attr]
            # Only write the attribute if it's True/1
            if value == 0 or value == False or value == [] or value == "[]":
                prob = 0
            elif pd.isna(value):
                prob = 0
            elif attr in attribute_limits.keys():
                if value >= attribute_limits[attr][1]:
                    prob = 1
                elif value <= attribute_limits[attr][0]:
                    prob = 0
                else:
                    prob = 0.5
            elif attr == "yoe":
                if value > 10:
                    prob = 1
                else:
                    prob = 0
            else:
                prob = 1
            

            f.write(f"{prob}::{attr}.\n")
    
        # First copy the base problog program
    base_program_path = f'problog_program_base_{iteration_number}.pl'
    with open(base_program_path, 'r') as base_file:
        base_content = base_file.read()
    
    # Open program file in append mode since we'll add more content after this
    with open(program_file, 'a') as f:
        f.write(base_content)
       