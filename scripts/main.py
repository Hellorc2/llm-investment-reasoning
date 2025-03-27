import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from C_insight_generation import analyze_founder, save_result
from Ea_llm_reasoning import analyze_insights, analyze_insights_in_groups
from F_document_generation import generate_logical_criteria_doc
from Fb_problog_generation import generate_problog_program
from Fc_bayesian_network import create_bayesian_network, display_bayesian_network, print_network_structure
from Fd_GBM import train_decision_tree, get_updated_rows
from datetime import datetime
from Z_data_utils import get_n_filtered_rows, get_n_random_rows_and_split
from typing import List, Tuple


def main():
    # Get a random founder's data (1 successful and 1 failed case)
    founder_data = pd.read_csv('selected_rows.csv')
    
    # Create empty dataframe to store results
    results_df = pd.DataFrame(columns=['founder', 'insight', 'success'])
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Counter for saving results
    processed_count = 0
    
    # Analyze each founder
    for index, row in founder_data.iterrows():
        # Combine LinkedIn and CB data for analysisanalysis
        combined_profile = f"LinkedIn Data: {row['cleaned_founder_linkedin_data']}\n\nCrunchbase Data: {row['cleaned_founder_cb_data']}"
        result = analyze_founder(combined_profile, row['success'], model="openai")
        
        # Create a timestamp for the session
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save the result
        save_result(timestamp, result)
        
        # Add to results DataFrame using concat
        new_row = pd.DataFrame({
            'founder': [row['founder_name']],
            'insight': [result],
            'success': [row['success']]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        
        print(result)
        
        # Increment counter and save results if we've processed 5 founders
        processed_count += 1
        if processed_count % 5 == 0:
            # Save intermediate results
            output_path = os.path.join(results_dir, f'founder_insights_batch_{processed_count//5}.csv')
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to: {output_path}")
    
    # Save final results
    final_output_path = os.path.join(results_dir, 'founder_insights_final.csv')
    results_df.to_csv(final_output_path, index=False)
    print(f"\nFinal results saved to: {final_output_path}")

def insight_analysis():
    analysis_results = analyze_insights_in_groups('results/founder_insights_final.csv', model="deepseek")
    print(analysis_results)

def generate_document():
    document = generate_logical_criteria_doc(model="deepseek")
    print(document)

def generate_program():
    program = generate_problog_program(model="deepseek")
    print(program)

def generate_bayesian_network():
    model = create_bayesian_network()
    print_network_structure(model)
    display_bayesian_network(model)

def generate_xgboost_model():
    # First get and save the updated rows
    df = get_updated_rows(columns=[
        'professional_athlete', 'childhood_entrepreneurship', 'competitions', 'ten_thousand_hours_of_mastery',
        'languages', 'perseverance', 'risk_tolerance', 'vision', 'adaptability', 'personal_branding',
        'education_level', 'education_institution', 'education_field_of_study', 'education_international_experience',
        'education_extracurricular_involvement', 'education_awards_and_honors', 'big_leadership', 'nasdaq_leadership',
        'number_of_leadership_roles', 'being_lead_of_nonprofits', 'number_of_roles', 'number_of_companies', 'industry_achievements',
        'big_company_experience', 'nasdaq_company_experience', 'big_tech_experience', 'google_experience', 'facebook_meta_experience',
        'microsoft_experience', 'amazon_experience', 'apple_experience', 'career_growth', 'moving_around',
        'international_work_experience', 'worked_at_military', 'big_tech_position', 'worked_at_consultancy', 'worked_at_bank',
        'press_media_coverage_count', 'vc_experience', 'angel_experience', 'quant_experience',
        'board_advisor_roles', 'tier_1_vc_experience', 'startup_experience', 'ceo_experience', 'investor_quality_prior_startup',
        'previous_startup_funding_experience', 'ipo_experience', 'num_acquisitions', 'domain_expertise', 'skill_relevance', 'yoe'
    ])
    
    # Only proceed with training if we successfully got the data
    if df is not None:
        train_decision_tree()
    else:
        print("Failed to process data. Please check the input files and columns.")

generate_xgboost_model()


