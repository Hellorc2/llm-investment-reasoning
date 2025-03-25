import pandas as pd
from typing import List
import os

# Get the parent directory path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read the founder scores CSV file
try:
    founder_df = pd.read_csv(os.path.join(parent_dir, 'founder_data.csv'))
except FileNotFoundError:
    print("Error: founder_data.csv file not found in the parent directory")
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

def get_n_random_rows_and_split(n: int, columns: List[str], output_selected: str = "selected_rows.csv", output_remaining: str = "remaining_rows.csv") -> pd.DataFrame:
    """
    Returns n randomly selected rows of founder data with specified columns and splits data into two CSV files
    
    Args:
        n: Number of rows to randomly select
        columns: List of column names to include
        output_selected: Filename for CSV containing selected rows (default: selected_rows.csv)
        output_remaining: Filename for CSV containing remaining rows (default: remaining_rows.csv)
            
    Returns:
        DataFrame containing n random rows with only the specified columns
        Or error message if any requested column does not exist
    """
    # Check if DataFrame is empty
    if founder_df.empty:
        return "Error: No data available. Please check if the CSV file exists and contains data."
            
    # Check if all requested columns exist in the data
    invalid_columns = [col for col in columns if col not in founder_df.columns]
        
    if invalid_columns:
        return f"Error: The following columns do not exist: {', '.join(invalid_columns)}"

    # Check if n is larger than total number of rows
    if n > len(founder_df):
        return f"Error: Requested {n} rows but only {len(founder_df)} rows exist in the data"
            
    # Randomly sample n rows
    selected_rows = founder_df[columns].sample(n=n, random_state=42)
    
    # Get the indices of selected rows
    selected_indices = selected_rows.index
    
    # Get remaining rows
    remaining_rows = founder_df[columns].drop(selected_indices)
    
    # Save to CSV files
    try:
        selected_rows.to_csv(os.path.join(parent_dir, output_selected), index=False)
        remaining_rows.to_csv(os.path.join(parent_dir, output_remaining), index=False)
    except Exception as e:
        print(f"Warning: Error saving CSV files: {str(e)}")
    
    return selected_rows
