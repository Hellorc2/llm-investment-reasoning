import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd


from datetime import datetime
from Z_data_utils import get_n_filtered_rows, get_n_random_rows_and_split
from typing import List, Tuple


def generate_insights(model="deepseek", iterative = False, iterative_index = 0):
    from C_insight_generation import analyze_founder, save_result

    if iterative:
        founder_data = pd.read_csv(f'iterative_train_data/train_batch_{iterative_index:03d}.csv')
    else:
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
        
        # Increment counter and save results if we've processed 10 founders
        processed_count += 1
        if processed_count % 10 == 0:
            # Save intermediate results
            output_path = os.path.join(results_dir, f'founder_insights_batch_{processed_count//10}.csv')
            results_df.to_csv(output_path, index=False)
            print(f"\nIntermediate results saved to: {output_path}")
    
    # Save final results
    if iterative:
        final_output_path = os.path.join(results_dir, f'founder_insights_final_{iterative_index:03d}.csv')
    else:
        final_output_path = os.path.join(results_dir, 'founder_insights_final.csv')
    results_df.to_csv(final_output_path, index=False)
    print(f"\nFinal results saved to: {final_output_path}")

def split_training_data(batch_size=50):
    # Create iterative_train_data directory if it doesn't exist
    data_dir = 'iterative_train_data'
    os.makedirs(data_dir, exist_ok=True)

    # Read the training data
    train_data = pd.read_csv('train_data.csv')
    
    # Split data into successful and unsuccessful
    successful = train_data[train_data['success'] == True]
    unsuccessful = train_data[train_data['success'] == False]
    
    # Calculate number of complete batches
    total_rows = len(train_data)
    num_complete_batches = total_rows // batch_size
    remaining_rows = total_rows % batch_size
    
    # Create batches
    for i in range(num_complete_batches):
        # Sample 5 successful and 45 unsuccessful rows
        batch_successful = successful.sample(n=5)
        batch_unsuccessful = unsuccessful.sample(n=45)
        
        # Combine and shuffle
        batch = pd.concat([batch_successful, batch_unsuccessful])
        batch = batch.sample(frac=1).reset_index(drop=True)
        
        batch.to_csv(os.path.join(data_dir, f'train_batch_{i:03d}.csv'), index=False)
    
    # Handle remaining rows as the last batch if any
    if remaining_rows > 0:
        # Calculate proportions for remaining rows to maintain roughly same ratio
        n_successful = max(1, int(remaining_rows * 0.1))  # At least 1 successful
        n_unsuccessful = remaining_rows - n_successful
        
        batch_successful = successful.sample(n=n_successful)
        batch_unsuccessful = unsuccessful.sample(n=n_unsuccessful)
        
        last_batch = pd.concat([batch_successful, batch_unsuccessful])
        last_batch = last_batch.sample(frac=1).reset_index(drop=True)
        last_batch.to_csv(os.path.join(data_dir, f'train_batch_{num_complete_batches:03d}.csv'), index=False)
    
    print(f"Split {total_rows} rows into {num_complete_batches + (1 if remaining_rows > 0 else 0)} batches")
    print(f"Batch size: {batch_size} (5 successful, 45 unsuccessful)")
    print(f"Last batch size: {remaining_rows if remaining_rows > 0 else batch_size}")
    print(f"Batches saved in: {data_dir}")


def insight_analysis():
    from Ea_llm_reasoning import analyze_insights, analyze_insights_in_groups, logical_statements_to_csv
    analysis_results = analyze_insights_in_groups('results/founder_insights_final.csv', model="deepseek")
    logical_statements_to_csv(model="deepseek")
    print(analysis_results)

def generate_document():
    from F_document_generation import generate_logical_criteria_doc
    document = generate_logical_criteria_doc(model="deepseek")
    print(document)

def generate_program():
    from Fb_problog_generation import generate_problog_program
    program = generate_problog_program(model="deepseek")
    print(program)

def raw_probability_from_logical_statements(founders_data_sample, iterative_index = 0):
    from arm import calculate_success_probability, calculate_failure_probability
    
    with open(f'logical_statements_preprocessed_{iterative_index:03d}.txt', 'r') as input_file, \
            open(f'logical_statements_polished_{iterative_index:03d}.txt', 'w') as output_file:
        for line in input_file:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(',')
                if parts[0] == '1':
                    # For success rules, call calculate_success_probability
                    # Sample 1000 rows with fixed seed for reproducibility
                    success_probability, len_filtered_data = calculate_success_probability(parts[1:-1], founders_data_sample)
                    if len_filtered_data < 10:
                        result = f"Success rule: {parts[1:-1]}, probability: not enough samples"
                    else:
                        result = f"Success rule: {parts[1:-1]}, probability: {success_probability/100:.2f}"
                    print(result)
                    output_file.write(result + '\n')
                elif parts[0] == '0': 
                    # For failure rules, call calculate_failure_probability
                    failure_probability, len_filtered_data = calculate_failure_probability(parts[1:-1], founders_data_sample)
                    if len_filtered_data < 10:
                        result = f"Failure rule: {parts[1:-1]}, probability: not enough samples"
                    else:
                        result = f"Failure rule: {parts[1:-1]}, probability: {(100-failure_probability)/100:.2f}"
                    print(result)
                    output_file.write(result + '\n')

def reflect_logical_statement(model="deepseek", iterative_index = 0):
    # Read the polished logical statements
    with open(f'logical_statements_polished_{iterative_index:03d}.txt', 'r') as f:
        polished_statements = f.read()
    
    with open(f'logical_statements_preprocessed_{iterative_index:03d}.txt', 'r') as f:
        logical_statements = f.read()

    # Prepare prompts for LLM analysis
    system_prompt = """You are an vc analyst analyzing startup success patterns. You had intuition about the success of a startup based on the founder's 
    attributes. Now you want to see if the probabilities you assigned to the rules are consistent with probability of success calculated from the data."""

    user_prompt = f"""Here are the original logical rules you generated based on your intuition:

    {logical_statements}

    Now I will show you the same rules with the probabilities calculated from the data:

    {polished_statements}

    Please reflect on your intuition by comparing the probabilities calculated from the data with the probabilities you assigned to the rules. 
    If the probabilities are not consistent, try to understand why and adjust your intuition. If the probability says "not enough samples", that means
    that the rule is satisfied very rarely, so you might have to think about whether your original logic came from an outliner. You should be particularly
    worried about success rule that have lower probability than a failure rule and vice versa - consider swapping out the rule entirely. Your end goal 
    is to produce a modified version of the original rules that are more accurate. Return me the modified rules in the same format as the original rules.
    Note that you are free to add, delete or modify the original rules."""

    # Get LLM response
    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    analysis = get_llm_response(system_prompt, user_prompt)
    print("\nLLM Analysis of Probability Patterns:")
    print("------------------------------------")
    print(analysis)

    # Write the LLM's reflected analysis to a new file
    with open(f'logical_statements_reflected_{iterative_index:03d}.txt', 'w') as f:
        f.write(analysis)

    return analysis



def iterative_training_step(model="deepseek", iterative_index=0):

    founders_data_sample = pd.read_csv('founder_data_ml.csv').sample(n=3000)

    if iterative_index == 0:
        generate_insights(iterative=False)
        insight_analysis()
    else:
        raw_probability_from_logical_statements(founders_data_sample, iterative_index = iterative_index)
        reflect_logical_statement(iterative_index = iterative_index)
        generate_new_insights(iterative_index = iterative_index)
    
    save_iterative_results(iterative_index = iterative_index)

def save_iterative_results(iterative_index):
    """
    Save logical statements from each iteration into a consolidated file.
    """
    # Create directory if it doesn't exist
    os.makedirs('iterative_training_results', exist_ok=True)
    
    # Read the logical statement files
    with open('logical_statements.txt', 'r') as f:
        logical = f.read()
    with open('logical_statements_polished.txt', 'r') as f:
        polished = f.read()
    with open('logical_statements_reflected.txt', 'r') as f:
        reflected = f.read()
    with open('logical_statements_preprocessed.txt', 'r') as f:
        preprocessed = f.read()
        
    # Write to consolidated file with iteration labels
    # Save each type of logical statements to separate files
    logical_path = os.path.join('iterative_training_results', f'iteration_{iterative_index:03d}_logical.txt')
    polished_path = os.path.join('iterative_training_results', f'iteration_{iterative_index:03d}_polished.txt')
    reflected_path = os.path.join('iterative_training_results', f'iteration_{iterative_index:03d}_reflected.txt')
    preprocessed_path = os.path.join('iterative_training_results', f'iteration_{iterative_index:03d}_preprocessed.txt')

    with open(logical_path, 'w') as f:
        f.write(f"=== Iteration {iterative_index} Logical Statements ===\n\n")
        f.write(logical)

    with open(polished_path, 'w') as f:
        f.write(f"=== Iteration {iterative_index} Polished Statements ===\n\n")
        f.write(polished)

    with open(reflected_path, 'w') as f:
        f.write(f"=== Iteration {iterative_index} Reflected Statements ===\n\n")
        f.write(reflected)
        
    with open(preprocessed_path, 'w') as f:
        f.write(f"=== Iteration {iterative_index} Preprocessed Statements ===\n\n")
        f.write(preprocessed)

def generate_new_insights(iterative_index):
    from C_insight_generation import analyze_founder, save_result
    from Ea_llm_reasoning import analyze_insights_in_groups, logical_statements_to_csv


    generate_insights(iterative=True, iterative_index=iterative_index)

    # Read the polished logical statements from the last iteration
    reflected_path = os.path.join('iterative_training_results', f'iteration_{iterative_index-1:03d}_reflected.txt')
    previous_analysis = ""
    if os.path.exists(reflected_path):
        with open(reflected_path, 'r') as f:
            previous_analysis = f.read()

    analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek", iterative=True, previous_analysis = previous_analysis)
    logical_statements_to_csv(model="deepseek")

def iterative_training():
    for i in range(0, 100):
        iterative_training_step(iterative_index=i)
        print(f"Iteration {i} complete")
        

def generate_bayesian_network():
    from Fc_bayesian_network import create_bayesian_network, display_bayesian_network, print_network_structure
    model = create_bayesian_network()
    print_network_structure(model)
    display_bayesian_network(model)

def generate_xgboost_model():
    from Fd_GBM import train_decision_tree, get_updated_rows
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



iterative_training()