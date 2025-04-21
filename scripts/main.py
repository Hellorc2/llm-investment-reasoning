import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from predict import *
from evaluate import evaluate
from llm_reasoning import logical_statements_preprocess, modify_analysis_based_on_advice

# Set display options to show full content
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def analyze_single_founder(args):
    index, row, model = args
    from insight_generation import analyze_founder
    
    print(f"Process {os.getpid()} starting analysis of founder {row['founder_name']}")
    
    # Combine LinkedIn and CB data for analysis
    combined_profile = f"LinkedIn Data: {row['cleaned_founder_linkedin_data']}\n\nCrunchbase Data: {row['cleaned_founder_cb_data']}"
    result = analyze_founder(combined_profile, row['success'], model=model)
    
    # Create a timestamp for the session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save the result
    from scripts.insight_generation import save_result
    save_result(timestamp, result)
    
    print(f"Process {os.getpid()} completed analysis of founder {row['founder_name']}")
    
    return {
        'founder': row['founder_name'],
        'insight': result,
        'success': row['success']
    }

def generate_insights_series(model="deepseek", iterative = False, iterative_index = 0):
    from scripts.insight_generation import analyze_founder, save_result

    if iterative:
        founder_data = pd.read_csv(f'iterative_train_data/train_batch_{iterative_index:03d}.csv')
    else:
        founder_data = pd.read_csv('selected_rows.csv') 
    
    for index, row in founder_data.iterrows():
        analyze_founder(row['cleaned_founder_linkedin_data'], row['success'], model=model)

    

def generate_insights(model="deepseek", iterative = False, iterative_index = 0):
    from scripts.insight_generation import analyze_founder, save_result

    if iterative:
        founder_data = pd.read_csv(f'iterative_train_data/train_batch_{iterative_index:03d}.csv')
    else:
        # Get a random founder's data (1 successful and 1 failed case)
        founder_data = pd.read_csv('test_data_reduced_2.csv')
    
    print(f"\nStarting parallel processing of {len(founder_data)} founders")
    print(f"Main process ID: {os.getpid()}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a pool of workers
    num_processes = min(cpu_count(), len(founder_data))  # Use at most 4 processes
    print(f"Creating pool with {num_processes} worker processes")
    pool = Pool(processes=num_processes)
    
    try:
        # Prepare arguments for parallel processing
        args = [(idx, row, model) for idx, row in founder_data.iterrows()]
        
        print("\nStarting parallel founder analysis...")
        # Process founders in parallel
        results = pool.map(analyze_single_founder, args)
        print("\nAll founder analyses completed")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save final results
        if iterative:
            final_output_path = os.path.join(results_dir, f'founder_insights_final_{iterative_index:03d}.csv')
        else:
            final_output_path = os.path.join(results_dir, 'founder_insights_final_test_data_reduced_2.csv')
        results_df.to_csv(final_output_path, index=False)
        print(f"\nFinal results saved to: {final_output_path}")
        
    finally:
        print("\nClosing worker pool...")
        pool.close()
        pool.join()
        print("Worker pool closed")



def insight_analysis(iterative = False, iterative_index = 0):
    from scripts.llm_reasoning import analyze_insights_in_groups, logical_statements_preprocess
    if iterative:
        analysis_results = analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek")
    else:
        analysis_results = analyze_insights_in_groups('results/founder_insights_final.csv', model="deepseek")
    logical_statements_preprocess(model="deepseek")
    print(analysis_results)

def iterative_training(starting_from = 0, end_at = 9, model="deepseek"):
    for i in range(starting_from, end_at+1):
        iterative_training_step(iterative_index=i)
        print(f"Iteration {i} complete")

def iterative_training_step(iterative_index=0, model="deepseek"):

    founders_data_sample = pd.read_csv('training_data_ml.csv').sample(n=1000, random_state=iterative_index)

    # Save the sampled data for this iteration
    output_path = f'training_data_sample_ml.csv'
    founders_data_sample.to_csv(output_path, index=False)

    
    if iterative_index == 0:
        #generate_insights(iterative=True, iterative_index=0)
        insight_analysis(iterative=True, iterative_index=0)
        save_iterative_results_1(iterative_index)
    elif iterative_index % 5 == 4:
        generate_new_insights(iterative_index = iterative_index)
        logical_statements_preprocess(model="deepseek")
        save_iterative_results_1(iterative_index)
        evaluate(iterative_index = iterative_index)
        logical_statements_preprocess(input_file = f'iterative_training_results/iteration_{iterative_index:03d}_advice.txt',
                                      output_file = f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', model="deepseek")
    else:
        generate_new_insights(iterative_index = iterative_index)
        logical_statements_preprocess(model="deepseek")
        save_iterative_results_1(iterative_index)
    

    success_rule_hints, failure_rule_hints = raw_probability_from_logical_statements(founders_data_sample, iterative_index = iterative_index)
    reflect_logical_statement(iterative_index = iterative_index, success_rule_hints = success_rule_hints, failure_rule_hints = failure_rule_hints)
    save_iterative_results_2(iterative_index)


def raw_probability_from_logical_statements(founders_data_sample, iterative_index = 0):
    from arm import calculate_success_probability, calculate_failure_probability, arm_success, arm_failure
    
    success_rule_hints = arm_success(founders_data_sample, feature_combination = 2, min_sample = 10, random_sample_size = 2)
    failure_rule_hints = arm_failure(founders_data_sample, feature_combination = 2, min_sample = 90, random_sample_size = 1)
    print(success_rule_hints)
    print(failure_rule_hints)
    with open(f'logical_statements/logical_statements_preprocessed.txt', 'r') as input_file, \
            open(f'logical_statements/logical_statements_calibrated.txt', 'w') as output_file:
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
                    output_file.write(result + '\n')
                elif parts[0] == '0': 
                    # For failure rules, call calculate_failure_probability
                    failure_probability, len_filtered_data = calculate_failure_probability(parts[1:-1], founders_data_sample)
                    if len_filtered_data < 90:
                        result = f"Failure rule: {parts[1:-1]}, probability: not enough samples"
                    else:
                        result = f"Failure rule: {parts[1:-1]}, probability: {failure_probability/100:.2f}"
                    output_file.write(result + '\n')
        output_file.write(f"Success rule hints: {success_rule_hints}\n")
        output_file.write(f"Failure rule hints: {failure_rule_hints}\n")
    return success_rule_hints, failure_rule_hints


def reflect_logical_statement(model="deepseek", success_rule_hints = "", failure_rule_hints = "", iterative_index = 0):
    # Read the polished logical statements
    with open(f'logical_statements/logical_statements_calibrated.txt', 'r') as f:
        calibrated_statements = f.read()
    
    with open(f'logical_statements/logical_statements_preprocessed.txt', 'r') as f:
        logical_statements = f.read()

    
    # Prepare prompts for LLM analysis
    system_prompt = """You are an vc analyst analyzing startup success patterns. You had intuition about the success of a startup based on the founder's 
    attributes. Now you want to see if the probabilities you assigned to the rules are consistent with probability of success calculated from the data."""

    user_prompt = f"""Here are the original logical rules you generated based on your intuition:

    {logical_statements}

    Now I will show you the same rules with the probabilities calculated from the data:

    {calibrated_statements}
    
    Also, you are optionally given one or few high-probability success rules discovered from the data:

    {success_rule_hints}

    and one high-probability failure rule discovered from the data:

    {failure_rule_hints};

    consider incorporating them into your rules, but make sure it fits the whole policy. Also, you MUST make sure to take these features in the right format -
    if a feature ends with _0 or _False, remove this part from the feature and add a not_ at the beginning of the feature. If a feature ends with
    _positive_number or _True, just remove this part.

    Please reflect on your intuition by comparing the probabilities calculated from the data with the probabilities you assigned to the rules. 
    If the probabilities are not consistent, try to understand why and adjust your intuition. If the probability says "not enough samples", that means
    that the rule is satisfied very rarely, so you might have to think about whether your original logic came from an outliner. If the success probability 
    was too low (random success probability in the dataset was 0.1, so anything less than 0.11 is low) or the failure probability was too low (random 
    failure probability was 0.9), consider removing them. Your end goal 
    is to produce a modified version of the original rules that are more accurate. 
    
    Note that you are free to delete or modify the original rules, but not allowed to add new rules. Return me the modified rules in the same format as the original rules,
     and double check you have made all the deletions you wished to make. """

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
    with open(f'logical_statements/logical_statements_reflected.txt', 'w') as f:
        f.write(analysis)

    return analysis

def save_iterative_results_1(iterative_index):
    # Create directory if it doesn't exist
    os.makedirs('iterative_training_results', exist_ok=True)

    # Save logical statements
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_logical_statements.txt', 'w') as f:
        with open('logical_statements/logical_statements.txt', 'r') as source:
            f.write(source.read())

    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'w') as f:
        with open('logical_statements/logical_statements_preprocessed.txt', 'r') as source:
            f.write(source.read())



def save_iterative_results_2(iterative_index):
    os.makedirs('iterative_training_results', exist_ok=True)
        # Save polished statements
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_calibrated.txt', 'w') as f:
        with open('logical_statements/logical_statements_calibrated.txt', 'r') as source:
            f.write(source.read())

    # Save reflected analysis
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_reflected.txt', 'w') as f:
        with open('logical_statements/logical_statements_reflected.txt', 'r') as source:
            f.write(source.read())

def generate_new_insights(iterative_index):
    from insight_generation import analyze_founder, save_result
    from llm_reasoning import analyze_insights_in_groups, logical_statements_preprocess


    #generate_insights(iterative=True, iterative_index=iterative_index)

    # Read the polished logical statements from the last iteration
    reflected_path = os.path.join('iterative_training_results', f'iteration_{iterative_index-1:03d}_reflected.txt')
    previous_analysis = ""
    if os.path.exists(reflected_path):
        with open(reflected_path, 'r') as f:
            previous_analysis = f.read()

    analysis_result = analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek", iterative=True, previous_analysis = previous_analysis)
    print(analysis_result)


if __name__ == "__main__":
    iterative_training(starting_from = 0, end_at = 9, model="deepseek")