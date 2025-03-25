import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from C_insight_generation import analyze_founder, save_result
from Ea_llm_reasoning import analyze_insights
from F_document_generation import generate_logical_criteria_doc
from datetime import datetime
from data_utils import founder_df, get_n_filtered_rows
from typing import List, Tuple


def main():
    # Get a random founder's data
    founder_data = get_n_filtered_rows(10, ['founder_name', 'cleaned_founder_linkedin_data', 'success'])
    
    # Create empty dataframe to store results
    results_df = pd.DataFrame(columns=['founder', 'insight', 'success'])
    
    # Analyze each founder
    for index, row in founder_data.iterrows():
        result = analyze_founder(row['cleaned_founder_linkedin_data'], row['success'])
        
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
    
    # Save the results to a CSV file
    results_df.to_csv('founder_insights.csv', index=False)
    
    # Generate documentation
    generate_documentation(results_df)

def insight_analysis():
    analysis_results = analyze_insights()
    print(analysis_results)

def generate_document():
    document = generate_logical_criteria_doc()
    print(document)

generate_document()