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

def get_n_random_rows_and_split(num_success: int, num_failure: int, columns: List[str], output_selected: str = "selected_rows.csv", output_remaining: str = "remaining_rows.csv") -> pd.DataFrame:
    """
    Returns randomly selected rows of founder data with specified columns, split by success/failure,
    and saves the remaining data to CSV files
    
    Args:
        num_success: Number of successful cases to randomly select
        num_failure: Number of failed cases to randomly select
        columns: List of column names to include
        output_selected: Filename for CSV containing selected rows (default: selected_rows.csv)
        output_remaining: Filename for CSV containing remaining rows (default: remaining_rows.csv)
            
    Returns:
        DataFrame containing the randomly selected rows with only the specified columns
        Or error message if any requested column does not exist
    """
    # Check if DataFrame is empty
    if founder_df.empty:
        return "Error: No data available. Please check if the CSV file exists and contains data."
            
    # Check if all requested columns exist in the data
    invalid_columns = [col for col in columns if col not in founder_df.columns]
        
    if invalid_columns:
        return f"Error: The following columns do not exist: {', '.join(invalid_columns)}"
    
    # Split data by success
    success_data = founder_df[founder_df['success'] == True]
    failure_data = founder_df[founder_df['success'] == False]
    
    # Check if we have enough data in each category
    if len(success_data) < num_success:
        return f"Error: Only {len(success_data)} successful cases available, but {num_success} requested"
    if len(failure_data) < num_failure:
        return f"Error: Only {len(failure_data)} failed cases available, but {num_failure} requested"
            
    # Randomly select rows from each category
    selected_success = success_data[columns].sample(n=num_success, random_state=42)
    selected_failure = failure_data[columns].sample(n=num_failure, random_state=42)
    
    # Combine selected rows
    selected_rows = pd.concat([selected_success, selected_failure])
    
    # Get remaining rows (excluding selected ones)
    remaining_rows = founder_df[~founder_df.index.isin(selected_rows.index)]
    
    # Save selected and remaining rows to CSV files
    selected_rows.to_csv(output_selected, index=False)
    remaining_rows.to_csv(output_remaining, index=False)
    
    return selected_rows

# Test the data loading and filtering
if __name__ == "__main__":
    print(get_n_filtered_rows(5, ['cleaned_founder_linkedin_data', 'cleaned_founder_cb_data', 'success']))
