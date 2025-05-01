import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime
from evaluate import evaluate
from llm_reasoning import logical_statements_preprocess
from data_utils import *
import shutil
import time  # Add time module for sleep function

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

def iterative_training(starting_from = 0, end_at = 9, model="deepseek", exclude_features_success = [], exclude_features_failure = []):
    for i in range(starting_from, end_at+1):
        exclude_features_success, exclude_features_failure = iterative_training_step(iterative_index=i, model=model, exclude_features_success = exclude_features_success, exclude_features_failure = exclude_features_failure)
        print(f"Iteration {i} complete")

def iterative_training_step(iterative_index=0, model="deepseek", exclude_features_success = [], exclude_features_failure = []):

    founders_data_sample = pd.read_csv('training_data_ml.csv').sample(n=1000, random_state=iterative_index)

    # Save the sampled data for this iteration
    output_path = f'training_data_sample_ml.csv'
    founders_data_sample.to_csv(output_path, index=False)

    
    if iterative_index == 0:
        #generate_insights(iterative=True, iterative_index=0)
        insight_analysis(iterative=True, iterative_index=0)
        save_iterative_results_1(iterative_index)
    
    elif iterative_index % 5 == 0:
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
    

    success_rule_hints, failure_rule_hints, exclude_features_success, exclude_features_failure = raw_probability_from_logical_statements(founders_data_sample, iterative_index = iterative_index, 
                                                                                                                                         exclude_features_success = exclude_features_success, exclude_features_failure = exclude_features_failure)
    reflect_logical_statement(iterative_index = iterative_index, success_rule_hints = success_rule_hints, failure_rule_hints = failure_rule_hints)
    save_iterative_results_2(iterative_index)
    return exclude_features_success, exclude_features_failure


def raw_probability_from_logical_statements(founders_data_sample, iterative_index = 0, exclude_features_success = [], exclude_features_failure = []):
    from arm import calculate_success_probability, calculate_failure_probability, arm_success, arm_failure
    if iterative_index == 7:
        exclude_features_success = []
        exclude_features_failure = []
    success_rule_hints, exclude_features_success = arm_success(founders_data_sample, feature_combination = 2, min_sample = 10, random_sample_size = 2, exclude_features = exclude_features_success, exclude_features_threshold = 4)
    failure_rule_hints, exclude_features_failure = arm_failure(founders_data_sample, feature_combination = 2, min_sample = 90, random_sample_size = 1, exclude_features = exclude_features_failure, exclude_features_threshold = 4)
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
    return success_rule_hints, failure_rule_hints, exclude_features_success, exclude_features_failure


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
        
    analysis = get_llm_response(system_prompt, user_prompt, model = "deepseek-chat")
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

def run_cross_validation():
    for i in range(0,12):
        run_cross_validation_nth_fold(i)
        print(f"==========Fold {i} complete==========")
        time.sleep(600)

def prepare_cross_validation_data():
    for i in range(0,12):
        prepare_cross_validation_nth_fold(i)


def prepare_cross_validation_nth_fold(nth_fold):

    # Read the nth fold data
    train_data = pd.read_csv(f'cross_validation_data/fold_{nth_fold}/train_data.csv')
    val_data = pd.read_csv(f'cross_validation_data/fold_{nth_fold}/validation_data.csv')
    test_data = pd.read_csv(f'cross_validation_data/fold_{nth_fold}/test_data.csv')

    attributes_ml = [
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
        "previous_startup_funding_experience", "ipo_experience", "num_acquisitions", "domain_expertise", "skill_relevance", 'success'
    ]

    training_data_ml = get_n_filtered_rows(4000, attributes_ml, f'cross_validation_data/fold_{nth_fold}/train_data.csv')
    training_data_ml.to_csv(f'cross_validation_data/fold_{nth_fold}/training_data_ml.csv', index=False)

    # Take random 1100 rows from training data as iterative training data
    get_n_random_rows_and_split(110, 990, [],f'cross_validation_data/fold_{nth_fold}/train_data.csv', 
                                                          f'cross_validation_data/fold_{nth_fold}/iterative_training_data.csv',
                                                          f'cross_validation_data/fold_{nth_fold}/training_data_remaining.csv')
    split_training_data(successful_rows = 10, unsuccessful_rows = 90, train_data_csv = f'cross_validation_data/fold_{nth_fold}/iterative_training_data.csv', data_dir = f'cross_validation_data/fold_{nth_fold}/iterative_training_data')

    # Create results directory if it doesn't exist
    results_dir = os.path.join('cross_validation_data', f'fold_{nth_fold}', 'results')
    os.makedirs(results_dir, exist_ok=True)

    for i in range(0,11):
        train_batch_file = f'cross_validation_data/fold_{nth_fold}/iterative_training_data/train_batch_{i:03d}.csv'
        # Read the train batch file
        df = pd.read_csv(train_batch_file)
        
        # Create a new DataFrame with the required columns
        new_df = pd.DataFrame({
            'founder': df['founder_name'],
            'insight': df['insight'],
            'success': df['success']
        })
        # Save to a new CSV file
        output_file = f'cross_validation_data/fold_{nth_fold}/results/founder_insights_final_{i:03d}.csv'

        
        new_df.to_csv(output_file, index=False)

    get_n_random_rows_and_split(50, 450, [],f'cross_validation_data/fold_{nth_fold}/validation_data.csv', 
                                f'cross_validation_data/fold_{nth_fold}/validation_data_iterative_0.csv',
                                f'cross_validation_data/fold_{nth_fold}/validation_data_remaining.csv')
    get_n_random_rows_and_split(50, 450, [],f'cross_validation_data/fold_{nth_fold}/validation_data_remaining.csv', 
                                f'cross_validation_data/fold_{nth_fold}/validation_data_iterative_1.csv',
                                f'cross_validation_data/fold_{nth_fold}/validation_data_final.csv')
    
def load_files(nth_fold):
    copy_folder(f'cross_validation_data/fold_{nth_fold}/iterative_training_data', f'iterative_training_data')
    copy_folder(f'cross_validation_data/fold_{nth_fold}/results', f'results')
    copy_file(f'cross_validation_data/fold_{nth_fold}/training_data_ml.csv', f'training_data_ml.csv')
    copy_file(f'cross_validation_data/fold_{nth_fold}/validation_data_iterative_0.csv', f'validation_data_iterative_0.csv')
    copy_file(f'cross_validation_data/fold_{nth_fold}/validation_data_iterative_1.csv', f'validation_data_iterative_1.csv')
    copy_file(f'cross_validation_data/fold_{nth_fold}/validation_data_final.csv', f'validation_data_final.csv')
    copy_file(f'cross_validation_data/fold_{nth_fold}/test_data.csv', f'test_data.csv')


def save_files(nth_fold):
    copy_folder('predictions', f'cross_validation_data/fold_{nth_fold}/predictions')
    copy_folder('iterative_training_results', f'cross_validation_data/fold_{nth_fold}/iterative_training_results')
    copy_folder('predictions_iterative', f'cross_validation_data/fold_{nth_fold}/predictions_iterative')


def clean_up_files(nth_fold):
    # Clear problog_programs folder
    problog_dir = 'problog_programs'
    if os.path.exists(problog_dir):
        shutil.rmtree(problog_dir)
        os.makedirs(problog_dir)
        print(f"Cleared {problog_dir} directory")
    

def run_cross_validation_nth_fold(nth_fold):
    load_files(nth_fold)

    iterative_training(starting_from = 0, end_at = 10, model="deepseek", exclude_features_success = [], exclude_features_failure = [])

    # Predict validation data
    from predict import predict
    from prediction_analysis import get_best_models, get_best_model_clusters, prediction_analysis
    
    # Create empty CSV files with headers for validation results
    validation_results_path = f'cross_validation_data/fold_{nth_fold}/prediction_results_validation.csv'
    validation_cluster_results_path = f'cross_validation_data/fold_{nth_fold}/prediction_results_cluster_validation.csv'
    
    pd.DataFrame(columns=['best_models', 'best_success_thresholds', 'best_failure_thresholds']).to_csv(validation_results_path, index=False)
    pd.DataFrame(columns=['best_model_clusters', 'best_cluster_success_thresholds', 'best_cluster_failure_thresholds']).to_csv(validation_cluster_results_path, index=False)
    
    
    # Process validation data
    for i in range(0,11):
        predict(f'cross_validation_data/fold_{nth_fold}/validation_data_final.csv', iteration_index = i, iterative=False)
    
        best_models, best_success_thresholds, best_failure_thresholds = get_best_models([i], num_top_results = 1, f_score_parameter = 0.25)
        best_model_clusters, best_cluster_success_thresholds, best_cluster_failure_thresholds = get_best_model_clusters([i], num_top_results = 1, f_score_parameter = 0.25)

        # Write validation results for this iteration
        validation_result = pd.DataFrame([{
            'best_models': best_models,
            'best_success_thresholds': best_success_thresholds[i],
            'best_failure_thresholds': best_failure_thresholds[i],
        }])
        validation_result.to_csv(validation_results_path, mode='a', header=False, index=False)
        
        validation_cluster_result = pd.DataFrame([{
            'best_model_clusters': best_model_clusters,
            'best_cluster_success_thresholds': best_cluster_success_thresholds[i],
            'best_cluster_failure_thresholds': best_cluster_failure_thresholds[i],
        }])
        validation_cluster_result.to_csv(validation_cluster_results_path, mode='a', header=False, index=False)
        
        print(f"Completed validation iteration {i}. Pausing for 1 minute to let CPU cool down...")
        time.sleep(60)  # Pause for 1 minute
    
    # Read the validation results to get thresholds for test predictions
    results_df_validation = pd.read_csv(validation_results_path)
    results_cluster_df_validation = pd.read_csv(validation_cluster_results_path)
    
    best_success_thresholds = results_df_validation['best_success_thresholds'].values
    best_failure_thresholds = results_df_validation['best_failure_thresholds'].values
    best_cluster_success_thresholds = results_cluster_df_validation['best_cluster_success_thresholds'].values
    best_cluster_failure_thresholds = results_cluster_df_validation['best_cluster_failure_thresholds'].values

        # Create empty CSV files with headers for test results
    test_results_path = f'cross_validation_data/fold_{nth_fold}/prediction_results_test.csv'
    test_cluster_results_path = f'cross_validation_data/fold_{nth_fold}/prediction_results_cluster_test.csv'
    
    pd.DataFrame(columns=['iteration', 'precision', 'accuracy', 'recall', 'f_score_half', 'f_score_quarter', 'true_positives']).to_csv(test_results_path, index=False)
    pd.DataFrame(columns=['iteration', 'precision', 'accuracy', 'recall', 'f_score_half', 'f_score_quarter', 'true_positives']).to_csv(test_cluster_results_path, index=False)
    
    # Process test data
    for i in range(0,11):
        predict(f'cross_validation_data/fold_{nth_fold}/test_data.csv', iteration_index=i, iterative=True)
        precision, accuracy, recall, f_score_half, f_score_quarter, true_positives = prediction_analysis(i, best_success_thresholds[i], best_failure_thresholds[i], iterative=False)
        precision_cluster, accuracy_cluster, recall_cluster, f_score_half_cluster, f_score_quarter_cluster, true_positives_cluster = prediction_analysis(i, best_cluster_success_thresholds[i], best_cluster_failure_thresholds[i], iterative=False)
        
        # Write test results for this iteration
        test_result = pd.DataFrame([{
            'iteration': i,
            'precision': precision,
            'accuracy': accuracy,
            'recall': recall,
            'f_score_half': f_score_half,
            'f_score_quarter': f_score_quarter,
            'true_positives': true_positives
        }])
        test_result.to_csv(test_results_path, mode='a', header=False, index=False)
        
        test_cluster_result = pd.DataFrame([{
            'iteration': i,
            'precision': precision_cluster,
            'accuracy': accuracy_cluster,
            'recall': recall_cluster,
            'f_score_half': f_score_half_cluster,
            'f_score_quarter': f_score_quarter_cluster,
            'true_positives': true_positives_cluster
        }])
        test_cluster_result.to_csv(test_cluster_results_path, mode='a', header=False, index=False)
        
        print(f"Completed test iteration {i}. Pausing for 1 minute to let CPU cool down...")
        time.sleep(120)  # Pause for 1 minute

    save_files(nth_fold)
    clean_up_files(nth_fold)

if __name__ == "__main__":
    run_cross_validation()
    


