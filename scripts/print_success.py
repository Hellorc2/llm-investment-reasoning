import pandas as pd
import sys

def print_success_column(csv_file='test_data.csv'):
    """
    Print the success column from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Get the success column
        success_column = data['success']
        
        # Print the success column
        print("Success column values:")
        for i, value in enumerate(success_column.values):
            print(f"Row {i}: {value}")
            
        # Print summary statistics
        print("\nSummary:")
        print(f"Total rows: {len(success_column)}")
        print(f"Success (1): {sum(success_column == 1)}")
        print(f"Failure (0): {sum(success_column == 0)}")
        print(f"Success rate: {sum(success_column == 1) / len(success_column):.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Get command line arguments
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'test_data.csv'
    
    print_success_column(csv_file) 