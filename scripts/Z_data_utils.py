import pandas as pd
from typing import List
import os

def get_n_filtered_rows(n: int, columns: List[str], base_csv: str = 'founder_data.csv') -> pd.DataFrame:
    """
    Returns the first n rows of founder data, including only the specified columns
    
    Args:
        n: Number of rows to return
        columns: List of column names to include
        base_csv: Path to the base CSV file (default: founder_data.csv)
            
    Returns:
        DataFrame containing the first n rows with only the specified columns
        Or error message if any requested column does not exist
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read the founder scores CSV file
    try:
        founder_df = pd.read_csv(os.path.join(parent_dir, base_csv))
    except FileNotFoundError:
        return f"Error: {base_csv} file not found in the parent directory"
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
            
    # Check if all requested columns exist in the data
    invalid_columns = [col for col in columns if col not in founder_df.columns]
        
    if invalid_columns:
        return f"Error: The following columns do not exist: {', '.join(invalid_columns)}"
            
    # Extract first n rows with specified columns
    return founder_df[columns].head(n)

def get_n_random_rows_and_split(num_success: int, num_failure: int, columns: List[str], 
                              base_csv: str = 'founder_scores_3_25_reasearch.csv',
                              output_selected: str = "selected_rows.csv", 
                              output_remaining: str = "remaining_rows.csv") -> pd.DataFrame:
    """
    Returns randomly selected rows of founder data with specified columns, split by success/failure,
    and saves the remaining data to CSV files
    
    Args:
        num_success: Number of successful cases to randomly select
        num_failure: Number of failed cases to randomly select
        columns: List of column names to include
        base_csv: Path to the base CSV file (default: founder_data.csv)
        output_selected: Filename for CSV containing selected rows (default: selected_rows.csv)
        output_remaining: Filename for CSV containing remaining rows (default: remaining_rows.csv)
            
    Returns:
        DataFrame containing the randomly selected rows with only the specified columns
        Or error message if any requested column does not exist
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read the founder scores CSV file
    try:
        founder_df = pd.read_csv(os.path.join(parent_dir, base_csv))
    except FileNotFoundError:
        return f"Error: {base_csv} file not found in the parent directory"
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
    
    if columns == []:
        columns = founder_df.columns.tolist()
    else:
            
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
    selected_success = success_data[columns].sample(n=num_success)
    selected_failure = failure_data[columns].sample(n=num_failure)
    
    # Combine selected rows
    selected_rows = pd.concat([selected_success, selected_failure])
    
    # Get remaining rows (excluding selected ones)
    remaining_rows = founder_df[~founder_df.index.isin(selected_rows.index)]
    
    # Save selected and remaining rows to CSV files
    selected_rows.to_csv(output_selected, index=False)
    remaining_rows.to_csv(output_remaining, index=False)
    
    return selected_rows

def count_successful_founders(base_csv: str = 'founder_data.csv') -> dict:
    """
    Counts the number of successful and unsuccessful founders in the dataset
    
    Args:
        base_csv: Path to the base CSV file (default: founder_data.csv)
            
    Returns:
        Dictionary containing counts of successful and unsuccessful founders
        Or error message if file not found
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read the founder scores CSV file
    try:
        founder_df = pd.read_csv(os.path.join(parent_dir, base_csv))
    except FileNotFoundError:
        return f"Error: {base_csv} file not found in the parent directory"
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
    
    # Count successful and unsuccessful founders
    success_count = len(founder_df[founder_df['success'] == True])
    failure_count = len(founder_df[founder_df['success'] == False])
    total_count = len(founder_df)
    
    return {
        'total_founders': total_count,
        'successful_founders': success_count,
        'unsuccessful_founders': failure_count,
        'success_rate': f"{(success_count/total_count)*100:.2f}%"
    }

# Test the data loading and filtering
if __name__ == "__main__":
    print(get_n_filtered_rows(5, ['cleaned_founder_linkedin_data', 'cleaned_founder_cb_data', 'success']))
    print("\nFounder Statistics:")
    print(count_successful_founders())
