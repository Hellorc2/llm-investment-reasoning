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




def generate_problog_program(iteration_number, founder_info):

    #include the attributes

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

    rules = read_preprocessed_rules(iteration_number)
    with open('problog_program.pl', 'w') as f:
        for attr in attributes:
            # Get the value from founder_info dataframe row
            value = founder_info[attr]
            # Only write the attribute if it's True/1
            if value == 0 or value == False:
                prob = 0
            elif pd.isna(value):
                prob = 0
            else:
                prob = 1
            f.write(f"{prob}::{attr}.\n")
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
