import pandas as pd
from typing import List

# Read the founder scores CSV file
try:
    founder_df = pd.read_csv('founder_scores_3_25_reasearch.csv')
except FileNotFoundError:
    print("Error: founder_scores_3_25_reasearch.csv file not found")
    founder_df = pd.DataFrame()
except Exception as e:
    print(f"Error reading CSV file: {str(e)}")
    founder_df = pd.DataFrame()

def get_n_filtered_rows(n: int, columns: List[str]) -> pd.DataFrame:
    """
    Returns the first n rows of founder data, including only the specified columns
    
    Args:
        n: Number of rows to return
        columns: List of column names to include
            
    Returns:
        DataFrame containing the first n rows with only the specified columns
        Or error message if any requested column does not exist
    """
    # Check if DataFrame is empty
    if founder_df.empty:
        return "Error: No data available. Please check if the CSV file exists and contains data."
            
    # Check if all requested columns exist in the data
    invalid_columns = [col for col in columns if col not in founder_df.columns]
        
    if invalid_columns:
        return f"Error: The following columns do not exist: {', '.join(invalid_columns)}"
            
    # Extract first n rows with specified columns
    return founder_df[columns].head(n)