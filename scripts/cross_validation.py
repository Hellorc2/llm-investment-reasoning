import pandas as pd

def combine_results_into_single_file():
    import os
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
        combined_df.to_csv('results/founder_insights.csv', index=False)


def combine_iterative_results():
    import os
    from pathlib import Path

    # Define the directory containing the iterative results
    results_dir = Path('iterative_train_data')
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Read all CSV files from the iterative_train_data directory
    for file in results_dir.glob('*.csv'):
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all dataframes and save to a new CSV file
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_file = 'iterative_train_data/combined_data.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Combined results saved to {output_file}")

def create_founder_insights_csv():
    founders_df = pd.read_csv('iterative_train_data/combined_data.csv')
    insights_df = pd.read_csv('results/founder_insights.csv')

def merge_and_validate_results():
    # Read both files
    founder_insights = pd.read_csv('results/founder_insights.csv')
    combined_results = pd.read_csv('iterative_train_data/combined_data.csv')
    
    # Add founder_index column to both dataframes
    founder_insights['founder_index'] = range(len(founder_insights))
    combined_results['founder_index'] = range(len(combined_results))
    
    # Validate that founder profiles match
    print("\nValidating founder profiles match:")
    mismatches = 0
    for idx in range(len(founder_insights)):
        founder_profile1 = founder_insights.iloc[idx]['founder_profile']
        founder_profile2 = combined_results.iloc[idx]['founder_profile']
        
        if founder_profile1 != founder_profile2:
            print(f"Mismatch found at index {idx}")
            print(f"Founder insights profile: {founder_profile1[:100]}...")
            print(f"Combined results profile: {founder_profile2[:100]}...")
            mismatches += 1
    
    if mismatches == 0:
        print("All founder profiles match!")
    else:
        print(f"\nTotal mismatches found: {mismatches}")
        return None
    
    # Merge the dataframes based on founder_index
    merged_df = pd.merge(
        founder_insights,
        combined_results,
        on='founder_index',
        suffixes=('_founder', '_iterative')
    )
    
    # Save the merged dataframe
    output_file = 'results/merged_results.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged results saved to {output_file}")
    
    return merged_df






    






    # Define the directory containing the results

