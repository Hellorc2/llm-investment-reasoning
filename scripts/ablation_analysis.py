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
import json
import time
import re

def vanilla_deepseek(csv_path: str = 'cross_validation_data/fold_4/test_data.csv'):
    from llms.deepseek import get_llm_response

    founders_data = pd.read_csv(csv_path)
    system_prompt = "You are a VC anaylst trying to predict the success of a startup based some information about the founders."
    
    # Initialize predictions list and batch counter
    predictions = []
    batch_counter = 0
    batch_size = 500
    
    # Slice the DataFrame first, then iterate
    for index, founder_data in founders_data.iloc[500:].iterrows():
        founder_profile = json.loads(founder_data['cleaned_founder_linkedin_data'])
        founder_info = {
            'Industry': founder_profile['industry'],
            'Education': founder_profile['educations'],
            'Work Experience': founder_profile['jobs'],

        }
        
        system_prompt = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
        All founders under consideration are sourced from LinkedIn profiles of companies that have raised between $100K and $4M in funding. 
        A successful founder is defined as one whose company has achieved either an exit or IPO valued at over $500M."""
        user_prompt = f"Given the following founder {founder_info} please output only 'Yes' or 'No' corresponding to whether or not the founder will be successful."

        prediction = get_llm_response(system_prompt, user_prompt)
        print(f"{founder_data['founder_name']}: {prediction}")
        predictions.append(1 if prediction.lower() == 'yes' else 0)
        batch_counter += 1
        
        # Save intermediate results every 500 entries
        if batch_counter % batch_size == 0:
            # Create a temporary DataFrame with current predictions
            temp_df = founders_data.iloc[500:500+len(predictions)].copy()
            temp_df['prediction'] = predictions
            
            # Save to a temporary file with batch number
            batch_num = (batch_counter // batch_size) + 1  # Add 1 since we're starting from batch 2
            temp_df.to_csv(f'vanilla_deepseek_test_batch_{batch_num:03d}.csv', index=False)
            print(f"\nSaved intermediate results for batch {batch_num}")
    
    # Add final predictions to the DataFrame
    founders_data['prediction'] = predictions
    
    # Save final results
    output_file = 'vanilla_deepseek_test_batch_final.csv'
    founders_data.to_csv(output_file, index=False)
    print(f"Final results saved to {output_file}")



def vanilla_deepseek_evaluation(csv_path: str = 'vanilla_deepseek_test_batch_006_with_definitionv.csv'):
    predictions_df = pd.read_csv(csv_path)
    
    # Convert predictions and true labels to integers
    y_true = predictions_df['success'].astype(int)
    y_pred = predictions_df['prediction'].astype(int)
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    
    # Calculate precision (true positives / (true positives + false positives))
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    
    # Calculate recall (true positives / (true positives + false negatives))
    actual_positives = (y_true == 1).sum()
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    
    # Print results
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Number of true positives: {true_positives}")
    print(f"Number of false positives: {predicted_positives - true_positives}")
    
    return accuracy, precision, recall

def vanilla_gpt4o(csv_path: str = 'test_data_reduced.csv'):
    from llms.openai import get_llm_response

    founders_data = pd.read_csv(csv_path)
    
    # Initialize predictions list
    predictions = []
    
    for index, founder_data in founders_data.iterrows():
        founder_profile = json.loads(founder_data['cleaned_founder_linkedin_data'])
        founder_info = {
            'Industry': founder_profile['industry'],
            'Education': founder_profile['educations'],
            'Work Experience': founder_profile['jobs'],

        }
        
        system_prompt = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
        All founders under consideration are sourced from LinkedIn profiles of companies that have raised between $100K and $4M in funding. 
        A successful founder is defined as one whose company has achieved either an exit or IPO valued at over $500M."""
        user_prompt = f"Given the following founder {founder_info} please output only 'Yes' or 'No' corresponding to whether or not the founder will be successful."

        prediction = get_llm_response(system_prompt, user_prompt, temperature = 0.0)
        predictions.append(prediction)
        founder_name = founder_data['founder_name']
        print(f"{founder_name}: {prediction}")
    
    # Add predictions to the DataFrame
    founders_data['prediction'] = predictions
    
    # Save to CSV
    output_file = 'vanilla_openai_test_data_reduced_with_definition_restricted.csv'
    founders_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def predict_success(founder_data):
    founder_profile = json.loads(founder_data['cleaned_founder_linkedin_data'])
    founder_info = {
        'Industry': founder_profile['industry'],
        'Education': founder_profile['educations'],
        'Work Experience': founder_profile['jobs'],

    }
    
    system_prompt = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
    All founders under consideration are sourced from LinkedIn profiles of companies that have raised between $100K and $4M in funding. 
    A successful founder is defined as one whose company has achieved either an exit or IPO valued at over $500M."""
    user_prompt = f"Given the following founder {founder_info} please output only 'Yes' or 'No' corresponding to whether or not the founder will be successful."

    from llms.deepseek import get_llm_response
    prediction = get_llm_response(system_prompt, user_prompt)
    return 1 if prediction.lower() == 'yes' else 0







def predict_success_wrapper_with_policy(row):
    return predict_success_with_policy(row)

def predict_success_with_policy(founder_data):
    founder_profile = json.loads(founder_data['cleaned_founder_linkedin_data'])
    founder_info = {
        'Industry': founder_profile['industry'],
        'Education': founder_profile['educations'],
        'Work Experience': founder_profile['jobs'],

    }
    
    system_prompt = """You are an expert in venture capital tasked with identifying successful founders from their unsuccessful counterparts. 
    You are given a policy consisting of prediction rules, each in the format of success/failure, condition1, condition2, probability of success/failure. """
    user_prompt = f"Given the following policy: {policy}. Given the following founder {founder_info} please output only 'Yes' or 'No' corresponding to whether or not the founder will be successful. Please use only the given policy to make your prediction and do not use your own knowledge."

    from llms.deepseek import get_llm_response
    prediction = get_llm_response(system_prompt, user_prompt)
    return 1 if prediction.lower() == 'yes' else 0

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

tqdm.pandas(desc="Processing founders")
combined_df = pd.read_csv('cross_validation_data/fold_0/validation_data_final.csv')

# Read the policy file
with open('cross_validation_data/fold_0/iterative_training_results/iteration_000_preprocessed.txt', 'r') as f:
    policy = f.read()

for i in range(len(combined_df)):
    combined_df.loc[i, 'openai_policy_prediction'] = predict_success_with_policy(combined_df.loc[i])
    print(f"{combined_df.loc[i, 'founder_name']}: {combined_df.loc[i, 'openai_policy_prediction']}")
combined_df[['success', 'openai_policy_prediction']].to_csv('openai_policy_predictions_validation_data_final_0.csv', index=False)


"""
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(predict_success_wrapper_with_policy, row) for _, row in founders_to_process.iterrows()]
    
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing founders")):
        predictions.append(future.result())
        batch_counter += 1
        
        # Save intermediate results every 500 founders
        if batch_counter % batch_size == 0:
            # Create a temporary DataFrame with current predictions
            temp_df = combined_df.iloc[500:500+len(predictions)].copy()
            temp_df['deepseek_policy_prediction'] = predictions
            
            # Save to a temporary file with batch number
            batch_num = (batch_counter // batch_size) + 1  # Add 1 since we're starting from batch 2
            temp_df[['success', 'deepseek_policy_prediction']].to_csv(
                f'deepseek_policy_predictions_batch_{batch_num}.csv', 
                index=False
            )
            print(f"\nSaved intermediate results for batch {batch_num}")

# Add predictions to the full DataFrame, starting from index 500
combined_df.loc[500:, 'deepseek_policy_prediction'] = predictions

# Calculate metrics
true_positives = ((combined_df['success'] == 1) & (combined_df['deepseek_policy_prediction'] == 1)).sum()
false_positives = ((combined_df['success'] == 0) & (combined_df['deepseek_policy_prediction'] == 1)).sum()
false_negatives = ((combined_df['success'] == 1) & (combined_df['deepseek_policy_prediction'] == 0)).sum()

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f05_score = (1.25 * precision * recall) / (0.25 * precision + recall)
f025_score = (17/16 * precision * recall) / (1/16 * precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F0.5 Score: {f05_score:.4f}")
print(f"F0.25 Score: {f025_score:.4f}")

# Create a confusion matrix
confusion_matrix = pd.crosstab(combined_df['success'], combined_df['deepseek_policy_prediction'], rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)

# Save results to a CSV file
combined_df[['success', 'deepseek_policy_prediction']].to_csv('deepseek_policy_predictions_test_data_validation.csv', index=False)
"""


    
    
    
    
    
    

