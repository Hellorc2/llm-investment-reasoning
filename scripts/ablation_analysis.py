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

def vanilla_deepseek(csv_path: str = 'test_data_reduced.csv'):
    from llms.deepseek import get_llm_response

    founders_data = pd.read_csv(csv_path)
    system_prompt = "You are a VC anaylst trying to predict the success of a startup based some information about the founders."
    
    # Initialize predictions list
    predictions = []
    
    for index, row in founders_data.iterrows():
        founder_profile = row['founder_profile']
        user_prompt = f"""Given the following founder's background and startup idea:
             Founder Profile (LinkedIn followed by Crunchbase): {founder_profile}
        Analyse the founder's background carefully, and give a prediction on whether the startup was successful or not. Return me only the prediction,
        1 if the startup was successful, 0 if it was not."""
        
        insight = get_llm_response(system_prompt, user_prompt)
        predictions.append(insight)
    
    # Add predictions to the DataFrame
    founders_data['prediction'] = predictions
    
    # Save to CSV
    output_file = 'vanilla_deepseek_test_data_reduced.csv'
    founders_data.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def vanilla_deepseek_evaluation(csv_path: str = 'vanilla_deepseek_test_data_reduced.csv'):
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
    
    return accuracy, precision, recall



    
    
    
    
    
    

