import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

from problog.program import PrologString
from problog import get_evaluatable
    

from datetime import datetime
from Z_data_utils import get_n_filtered_rows, get_n_random_rows_and_split
from typing import List, Tuple
from G_generate_problog import generate_problog_program


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

def split_training_data(successful_rows = 5, unsuccessful_rows = 45):
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
    num_complete_batches = total_rows // (successful_rows + unsuccessful_rows)
    remaining_rows = total_rows % (successful_rows + unsuccessful_rows)
    
    # Create batches
    for i in range(num_complete_batches):
        # Sample 5 successful and 45 unsuccessful rows
        batch_successful = successful.sample(n=successful_rows)
        batch_unsuccessful = unsuccessful.sample(n=unsuccessful_rows)
        
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
    
    print(f"Batches saved in: {data_dir}")


def insight_analysis(iterative = False, iterative_index = 0):
    from Ea_llm_reasoning import analyze_insights, analyze_insights_in_groups, logical_statements_preprocess
    if iterative:
        analysis_results = analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek")
    else:
        analysis_results = analyze_insights_in_groups('results/founder_insights_final.csv', model="deepseek")
    logical_statements_preprocess(model="deepseek")
    print(analysis_results)


def raw_probability_from_logical_statements(founders_data_sample, iterative_index = 0):
    from arm import calculate_success_probability, calculate_failure_probability
    
    with open(f'logical_statements_preprocessed.txt', 'r') as input_file, \
            open(f'logical_statements_polished.txt', 'w') as output_file:
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
    with open(f'logical_statements_polished.txt', 'r') as f:
        polished_statements = f.read()
    
    with open(f'logical_statements_preprocessed.txt', 'r') as f:
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
    with open(f'logical_statements_reflected.txt', 'w') as f:
        f.write(analysis)

    return analysis

def save_iterative_results(iterative_index):
    # Create directory if it doesn't exist
    os.makedirs('iterative_training_results', exist_ok=True)

    # Save logical statements
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_logical_statements.txt', 'w') as f:
        with open('logical_statements.txt', 'r') as source:
            f.write(source.read())

    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'w') as f:
        with open('logical_statements_preprocessed.txt', 'r') as source:
            f.write(source.read())


    # Save polished statements
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_polished.txt', 'w') as f:
        with open('logical_statements_polished.txt', 'r') as source:
            f.write(source.read())

    # Save problog program

    # Save reflected analysis
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_reflected.txt', 'w') as f:
        with open('logical_statements_reflected.txt', 'r') as source:
            f.write(source.read())


def iterative_training_step(model="deepseek", iterative_index=0):

    founders_data_sample = pd.read_csv('founder_data_ml.csv').sample(n=1000)

    if iterative_index == 0:
        generate_insights(iterative=True, iterative_index=0)
        insight_analysis(iterative=True, iterative_index=0)
    else:
        generate_new_insights(iterative_index = iterative_index)


    raw_probability_from_logical_statements(founders_data_sample, iterative_index = iterative_index)
    reflect_logical_statement(iterative_index = iterative_index)
    save_iterative_results(iterative_index)

def generate_new_insights(iterative_index):
    from C_insight_generation import analyze_founder, save_result
    from Ea_llm_reasoning import analyze_insights_in_groups, logical_statements_preprocess


    generate_insights(iterative=True, iterative_index=iterative_index)

    # Read the polished logical statements from the last iteration
    reflected_path = os.path.join('iterative_training_results', f'iteration_{iterative_index-1:03d}_reflected.txt')
    previous_analysis = ""
    if os.path.exists(reflected_path):
        with open(reflected_path, 'r') as f:
            previous_analysis = f.read()

    analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek", iterative=True, previous_analysis = previous_analysis)
    logical_statements_preprocess(model="deepseek")

def iterative_training(starting_from = 0):
    for i in range(starting_from, 100):
        iterative_training_step(iterative_index=i)
        print(f"Iteration {i} complete")


def predict_success_of_founder(row_number, founder_info, iteration_index, threshold_success, threshold_failure):

    # Convert pandas Series to dictionary
    founder_dict = founder_info.to_dict()
    generate_problog_program(iteration_index, founder_dict)
    

    # Read the generated program
    with open('problog_program.pl', 'r') as f:
        program = f.read()
    
    # Create and evaluate the model using the 'approximate' engine
    model = PrologString(program)
    result = get_evaluatable().create_from(model).evaluate(engine='approximate')
    
    print(row_number, "Problog result:", result)  # Debug print to see the actual format
    
    # Handle empty result case
    if not result:
        print(f"Warning: Empty result for row {row_number}")
        return f'{row_number},0.0,1.0,{founder_info["success"]}\n'
    
    # Convert result to dictionary and get success probability
    result_dict = {str(k): float(v) for k, v in result.items()}
    success_prob = result_dict.get('success', 0.0)
    failure_prob = result_dict.get('failure', 1.0)
    
    is_success = founder_info['success']
    
    # Return the results instead of writing to file
    return f'{row_number},{success_prob},{failure_prob},{is_success}\n'


def manual_prediction_analysis(iteration_index, threshold_success = 0.3, threshold_failure = 1):
    predictions_df = pd.read_csv(f'prediction_{iteration_index}.csv')
    # Add a predictions column based on thresholds
    predictions_df['prediction'] = ((predictions_df['success_prob'] > threshold_success) & 
                                  (predictions_df['failure_prob'] < threshold_failure)).astype(int)
    
    # Calculate accuracy metrics
    total_predictions = len(predictions_df)
    correct_predictions = sum(predictions_df['prediction'] == predictions_df['is_success'])
    accuracy = correct_predictions / total_predictions

    # Calculate precision
    true_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 1))
    false_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 0))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    print(f"\nPrediction Summary for success threshold {threshold_success} and failure threshold {threshold_failure}:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Number of true positives: {true_positives}")
    print(f"Number of false positives: {false_positives}")
    """
    with open(f'prediction_summary_{iteration_index}.txt', 'a') as f:
        f.write(f"Prediction Summary for Iteration {iteration_index}\n")
        f.write(f"Success Threshold: {threshold_success}\n")
        f.write(f"Failure Threshold: {threshold_failure}\n")
        f.write(f"Total predictions: {total_predictions}\n")
        f.write(f"Correct predictions: {correct_predictions}\n") 
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.2%}\n")
        f.write(f"Number of true positives: {true_positives}\n")
        f.write(f"Number of false positives: {false_positives}\n")
    """
    
    return precision

        

def predict(csv_file, iteration_index, threshold_success = 0.2, threshold_failure = 0.999):
    # Read the data using pandas
    data = pd.read_csv(csv_file)
    # Clear prediction.csv before starting predictions
    import shutil
    try:
        shutil.copyfile(f'prediction_{iteration_index}.csv', f'prediction_copy_{iteration_index}.csv')
    except FileNotFoundError:
        pass  # Ignore if original file doesn't exist yet
    with open(f'prediction_{iteration_index}.csv', 'w') as f:
        f.write('row_number,success_prob,failure_prob,is_success\n')
    
    # Convert DataFrame to list of tuples for parallel processing
    from multiprocessing import Pool, cpu_count
    import itertools
    
    # Create list of (index, row) tuples
    rows_to_process = list(data.iterrows())
    
    # Create a pool of workers
    num_processes = cpu_count()  # Use number of CPU cores
    pool = Pool(processes=num_processes)
    
    # Process rows in parallel with batch saving
    batch_size = 80
    results = []
    for i in range(0, len(rows_to_process), batch_size):
        batch = rows_to_process[i:i + batch_size]
        batch_results = pool.starmap(predict_success_of_founder, 
                                   [(idx, row, iteration_index, threshold_success, threshold_failure) 
                                    for idx, row in batch])
        
        # Save this batch
        with open(f'prediction_{iteration_index}.csv', 'a') as f:
            f.writelines(batch_results)
        
        print(f"Processed and saved {min(i + batch_size, len(rows_to_process))} rows out of {len(rows_to_process)}")
    
    # Close the pool
    pool.close()
    pool.join()
    
    print("finished")


        
        

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

"""
import numpy as np
import matplotlib.pyplot as plt

# Store results for plotting
results = []
for iteration_index in [0,1,2,10,20]:
    for j in [0.99, 0.999, 0.9999]:
        for i in np.arange(0.05, 0.95, 0.05):
            precision = manual_prediction_analysis(iteration_index, i, j)
            results.append({
                'iteration': iteration_index,
                'threshold_success': i,
                'threshold_failure': j,
                'precision': precision
            })


# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Create three separate plots, one for each failure threshold
for j in [0.99, 0.999, 0.9999]:
    plt.figure(figsize=(12, 6))
    
    # Filter data for this failure threshold
    threshold_data = results_df[results_df['threshold_failure'] == j]
    
    # Plot lines for each iteration
    for iteration in threshold_data['iteration'].unique():
        iteration_data = threshold_data[threshold_data['iteration'] == iteration]
        plt.plot(iteration_data['threshold_success'], iteration_data['precision'], 
                 label=f'Iteration {iteration}', marker='o')
    
    plt.xlabel('Success Threshold')
    plt.ylabel('Precision')
    plt.title(f'Precision vs Success Threshold (Failure Threshold = {j})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'precision_analysis_failure_threshold_{j}.png')
    plt.close()

# Print summary statistics for each failure threshold
print("\nSummary Statistics by Failure Threshold:")
for j in [0.99, 0.999, 0.9999]:
    print(f"\nFailure Threshold = {j}")
    threshold_data = results_df[results_df['threshold_failure'] == j]
    print(threshold_data.groupby('iteration')['precision'].agg(['mean', 'max', 'min']))
"""

if __name__ == '__main__':
    predict(csv_file='test_data_reduced.csv', iteration_index=21)