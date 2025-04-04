import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from multiprocessing import Pool
from datetime import datetime
from predict import *


def analyze_single_founder(args):
    index, row, model = args
    from C_insight_generation import analyze_founder
    
    print(f"Process {os.getpid()} starting analysis of founder {row['founder_name']}")
    
    # Combine LinkedIn and CB data for analysis
    combined_profile = f"LinkedIn Data: {row['cleaned_founder_linkedin_data']}\n\nCrunchbase Data: {row['cleaned_founder_cb_data']}"
    result = analyze_founder(combined_profile, row['success'], model=model)
    
    # Create a timestamp for the session
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save the result
    from C_insight_generation import save_result
    save_result(timestamp, result)
    
    print(f"Process {os.getpid()} completed analysis of founder {row['founder_name']}")
    
    return {
        'founder': row['founder_name'],
        'insight': result,
        'success': row['success']
    }

def generate_insights_series(model="deepseek", iterative = False, iterative_index = 0):
    from C_insight_generation import analyze_founder, save_result

    if iterative:
        founder_data = pd.read_csv(f'iterative_train_data/train_batch_{iterative_index:03d}.csv')
    else:
        founder_data = pd.read_csv('selected_rows.csv') 
    
    for index, row in founder_data.iterrows():
        analyze_founder(row['cleaned_founder_linkedin_data'], row['success'], model=model)

    

def generate_insights(model="deepseek", iterative = False, iterative_index = 0):
    from C_insight_generation import analyze_founder, save_result

    if iterative:
        founder_data = pd.read_csv(f'iterative_train_data/train_batch_{iterative_index:03d}.csv')
    else:
        # Get a random founder's data (1 successful and 1 failed case)
        founder_data = pd.read_csv('selected_rows.csv')
    
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
            final_output_path = os.path.join(results_dir, 'founder_insights_final.csv')
        results_df.to_csv(final_output_path, index=False)
        print(f"\nFinal results saved to: {final_output_path}")
        
    finally:
        print("\nClosing worker pool...")
        pool.close()
        pool.join()
        print("Worker pool closed")



def insight_analysis(iterative = False, iterative_index = 0):
    from Ea_llm_reasoning import analyze_insights, analyze_insights_in_groups, logical_statements_preprocess
    if iterative:
        analysis_results = analyze_insights_in_groups(f'results/founder_insights_final_{iterative_index:03d}.csv', model="deepseek")
    else:
        analysis_results = analyze_insights_in_groups('results/founder_insights_final.csv', model="deepseek")
    logical_statements_preprocess(model="deepseek")
    print(analysis_results)

def iterative_training(starting_from = 0, model="deepseek"):
    for i in range(starting_from, 100):
        iterative_training_step(iterative_index=i)
        print(f"Iteration {i} complete")

def iterative_training_step(iterative_index=0, model="deepseek"):

    founders_data_sample = pd.read_csv('founder_data_ml.csv').sample(n=1000)

    if iterative_index == 0:
        generate_insights(iterative=True, iterative_index=0)
        insight_analysis(iterative=True, iterative_index=0)
    else:
        generate_new_insights(iterative_index = iterative_index)


    raw_probability_from_logical_statements(founders_data_sample, iterative_index = iterative_index)
    reflect_logical_statement(iterative_index = iterative_index)
    save_iterative_results(iterative_index)


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
                        result = f"Failure rule: {parts[1:-1]}, probability: {failure_probability/100:.2f}"
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
    Note that you are free to delete or modify the original rules, but not allowed to add new rules."""

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

from predict import plot_precision_analysis

plot_precision_analysis(iterations = [2,3,4,10,19,20])

    # Read all results files

