import pandas as pd
import sys
import ast

def print_language_column(csv_file='test_data.csv'):
    """
    Print entries with 9 languages from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Check if languages column exists
        if 'languages' not in data.columns:
            print("Error: 'languages' column not found in the dataset.")
            print("Available columns:", data.columns.tolist())
            return
            
        # Get the languages column and convert string representations of lists to actual lists
        languages_column = data['languages'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        
        # Print entries with 9 languages
        print("Founders who speak 9 languages:")
        found = False
        for i, (name, value) in enumerate(zip(data['founder_name'], languages_column)):
            if len(value) == 9:
                print(f"Row {i}: {name} - {value}")
                found = True
                
        if not found:
            print("No founders found who speak exactly 9 languages.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Get command line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_data.csv'
    
    print_language_column(csv_file) 