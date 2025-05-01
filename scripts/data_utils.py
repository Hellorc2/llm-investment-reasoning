import pandas as pd
from typing import List
import os
import shutil

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

def replace_language_column(base_csv) -> pd.DataFrame:
    """
    Replaces the language column with the language_cleaned column
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    # Read the founder data CSV file
    try:
        founder_df = pd.read_csv(os.path.join(parent_dir, base_csv))
    except FileNotFoundError:
        return f"Error: {base_csv} file not found in the parent directory"
        
    # Replace the language column with the language_cleaned column
    # Handle NaN values by checking if the value is a string before applying len()
    founder_df['languages'] = founder_df['languages'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

    # Save the updated DataFrame back to CSV
    try:
        founder_df.to_csv(os.path.join(parent_dir, base_csv), index=False)
        return founder_df
    except Exception as e:
        return f"Error saving CSV file: {str(e)}"


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

def split_training_data(successful_rows = 5, unsuccessful_rows = 45, train_data_csv = 'train_data.csv', data_dir = 'iterative_train_data'):
    # Create iterative_train_data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Read the training data
    train_data = pd.read_csv(train_data_csv)
    
    # Split data into successful and unsuccessful
    successful = train_data[train_data['success'] == True]
    unsuccessful = train_data[train_data['success'] == False]
    
    # Calculate number of complete batches
    num_successful_batches = len(successful) // successful_rows
    num_unsuccessful_batches = len(unsuccessful) // unsuccessful_rows
    num_complete_batches = min(num_successful_batches, num_unsuccessful_batches)
    
    # Create batches
    for i in range(num_complete_batches):
        # Get sequential slices of data
        start_success = i * successful_rows
        end_success = (i + 1) * successful_rows
        start_unsuccess = i * unsuccessful_rows
        end_unsuccess = (i + 1) * unsuccessful_rows
        
        batch_successful = successful.iloc[start_success:end_success]
        batch_unsuccessful = unsuccessful.iloc[start_unsuccess:end_unsuccess]
        
        # Combine and shuffle
        batch = pd.concat([batch_successful, batch_unsuccessful])
        batch = batch.sample(frac=1).reset_index(drop=True)
        
        batch.to_csv(os.path.join(data_dir, f'train_batch_{i:03d}.csv'), index=False)
    
    # Handle remaining rows as the last batch if any
    remaining_successful = successful.iloc[num_complete_batches * successful_rows:]
    remaining_unsuccessful = unsuccessful.iloc[num_complete_batches * unsuccessful_rows:]
    
    if len(remaining_successful) > 0 or len(remaining_unsuccessful) > 0:
        last_batch = pd.concat([remaining_successful, remaining_unsuccessful])
        last_batch = last_batch.sample(frac=1).reset_index(drop=True)
        last_batch.to_csv(os.path.join(data_dir, f'train_batch_{num_complete_batches:03d}.csv'), index=False)
    
    print(f"Batches saved in: {data_dir}")

def combine_test_batches():
    """
    Combines every four files in iterative_test_data directory and relabels them.
    Creates new files with combined data in the same directory, with sequential batch numbers.
    Deletes the original files after combining.
    """
    # Create or ensure directory exists
    data_dir = 'iterative_test_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Get all CSV files in the directory
    test_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    if len(test_files) < 4:
        print("Need at least 4 files to combine")
        return
    
    # Process files in groups of four
    for batch_idx in range(0, len(test_files) // 4):
        start_idx = batch_idx * 4
        if start_idx + 3 >= len(test_files):
            break  # Skip if we don't have a complete group of four
            
        # Read all four files
        files = [os.path.join(data_dir, test_files[start_idx + j]) for j in range(4)]
        dfs = [pd.read_csv(f) for f in files]
        
        # Combine the dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Relabel the batch numbers to be sequential
        combined_df['batch'] = batch_idx
        
        # Create new filename with original batch numbers
        batch_nums = [f.split('_')[2].split('.')[0] for f in test_files[start_idx:start_idx+4]]
        new_filename = f'test_batch_{"_".join(batch_nums)}.csv'
        
        # Save combined file
        combined_df.to_csv(os.path.join(data_dir, new_filename), index=False)
        
        # Delete original files
        for file in files:
            os.remove(file)
            print(f"Deleted {file}")
        
        print(f"Combined {', '.join(test_files[start_idx:start_idx+4])} into {new_filename} with batch number {batch_idx}")

def count_successful_in_test_batch(batch_number: int = 0) -> dict:
    """
    Counts the number of successful and unsuccessful founders in a specific test batch
    
    Args:
        batch_number: The batch number to analyze (default: 0)
            
    Returns:
        Dictionary containing counts of successful and unsuccessful founders
        Or error message if file not found
    """
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read the test batch CSV file
    try:
        batch_df = pd.read_csv(os.path.join(parent_dir, f'iterative_test_data/test_batch_{batch_number:03d}.csv'))
    except FileNotFoundError:
        return f"Error: test_batch_{batch_number:03d}.csv not found in iterative_test_data directory"
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
    
    # Count successful and unsuccessful founders
    success_count = len(batch_df[batch_df['success'] == True])
    failure_count = len(batch_df[batch_df['success'] == False])
    total_count = len(batch_df)
    
    return {
        'total_founders': total_count,
        'successful_founders': success_count,
        'unsuccessful_founders': failure_count,
        'success_rate': f"{(success_count/total_count)*100:.2f}%"
    }

def copy_folder(source_folder, destination_folder):
    """
    Copy an entire folder to another location.
    
    Args:
        source_folder (str): Path to the source folder
        destination_folder (str): Path to the destination folder
    """
    try:
        # Remove destination folder if it exists
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
        # Copy the folder
        shutil.copytree(source_folder, destination_folder)
        print(f"Successfully copied {source_folder} to {destination_folder}")
    except Exception as e:
        print(f"Error copying folder: {e}")

def copy_file(source_file, destination_file):
    """
    Copy a file to another location.
    """
    shutil.copy(source_file, destination_file)
    print(f"Successfully copied {source_file} to {destination_file}")


