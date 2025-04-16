def combine_results_into_single_file():
    import os
    import pandas as pd
    from pathlib import Path

    # Define the directory containing the results
    results_dir = Path('results')
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Read all CSV files from the results directory
    for file in results_dir.glob('*.csv'):
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all dataframes and save to a new CSV file
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv('results/results.csv', index=False)






    # Define the directory containing the results

