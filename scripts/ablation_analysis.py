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
import time  # Add this import

def vanilla_deepseek(csv_path: str = 'iterative_test_data/test_batch_006.csv'):
    from llms.deepseek import get_llm_response

    founders_data = pd.read_csv(csv_path)
    system_prompt = "You are a VC anaylst trying to predict the success of a startup based some information about the founders."
    
    # Initialize predictions list
    predictions = []
    
    for index, row in founders_data.iterrows():
        founder_name = row['founder_name']
        linkedin_profile = row['cleaned_founder_linkedin_data']
        crunchbase_profile = row['cleaned_founder_cb_data']
        combined_profile = f"LinkedIn Data: {row['cleaned_founder_linkedin_data']}\n\nCrunchbase Data: {row['cleaned_founder_cb_data']}"
        user_prompt = f"""Given the following founder's background and startup idea:
             Founder Profile (crunchbase): {combined_profile}
        Analyse the founder's background carefully, and give a prediction on whether the startup will be successful or not. A successful
        startup is defined as one which has achieved either an exit or IPO valued at over $500M or raised more than $500M in funding.
        At the start of your response, clearly state Prediction: followed by 1 if the startup was predicted to be successful, 0 if it was not. 
        DO NOT try to search up any of the specific companies or people or links from the internet!"""
        
        insight = get_llm_response("", user_prompt)
        print(f"{founder_name}: {insight}, actual success: {row['success']}")
        predictions.append(insight)
    
    # Add predictions to the DataFrame
    founders_data['prediction'] = predictions
    
    # Save to CSV
    output_file = 'vanilla_deepseek_test_batch_006_with_definitionv.csv'
    founders_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")



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
    system_prompt = "You are a VC anaylst trying to predict the success of a startup based some information about the founders."
    
    # Initialize predictions list
    predictions = []
    
    for index, row in founders_data.iterrows():
        founder_name = row['founder_name']
        combined_profile = f"LinkedIn Data: {row['cleaned_founder_linkedin_data']}\n\nCrunchbase Data: {row['cleaned_founder_cb_data']}"
        user_prompt = f"""Given the following founder's background and startup idea:
             Founder Profile (LinkedIn followed by Crunchbase): {combined_profile}
        Analyse the founder's background carefully, and give a prediction on whether the startup will be successful or not.  A successful
        startup is defined as one which has achieved either an exit or IPO valued at over $500M or raised more than $500M in funding.
        At the start of your response, clearly state Prediction: followed by 1 if the startup was predicted to be successful, 0 if it was not. 
        DO NOT try to search up any of the specific companies or people or links from the internet!"""
        
        insight = get_llm_response(system_prompt, user_prompt)
        print(f"{founder_name}: {insight}")
        predictions.append(insight)
    
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
    user_prompt = f"Given the following founder {founder_info} please output only 'Yes' or 'No' corresponding to whether or not the founder will be successful. DO NOT attempt to search up any of the specific companies or people or links from the internet!"

    from llms.deepseek import get_llm_response
    prediction = get_llm_response(system_prompt, user_prompt)
    return 1 if prediction.lower() == 'yes' else 0


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

tqdm.pandas(desc="Processing founders")
combined_df = pd.read_csv('test_data_validation.csv')

def predict_success_wrapper(row):
    return predict_success(row)


predictions = []

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(predict_success_wrapper, row) for _, row in combined_df.drop('success', axis=1).iterrows()]
    
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing founders")):
        predictions.append(future.result())

combined_df['gpt4_prediction'] = predictions

# Calculate metrics
true_positives = ((combined_df['success'] == 1) & (combined_df['gpt4_prediction'] == 1)).sum()
false_positives = ((combined_df['success'] == 0) & (combined_df['gpt4_prediction'] == 1)).sum()
false_negatives = ((combined_df['success'] == 1) & (combined_df['gpt4_prediction'] == 0)).sum()

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f05_score = (1.25 * precision * recall) / (0.25 * precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F0.5 Score: {f05_score:.4f}")

# Create a confusion matrix
confusion_matrix = pd.crosstab(combined_df['success'], combined_df['gpt4_prediction'], rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)

# Save results to a CSV file
combined_df[['success', 'gpt4_prediction']].to_csv('deepseek_predictions_test_data_validation.csv', index=False)









    
    
    
    
    
    

