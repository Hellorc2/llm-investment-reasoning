import pandas as pd
from data_utils import *
def combine_results_into_single_file():
    import os
    from pathlib import Path

    # Define the directory containing the results
    results_dir = Path('results')
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Read all CSV files from the results directory in sorted order
    for file in sorted(results_dir.glob('*.csv')):
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all dataframes and save to a new CSV file
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Add an index column to preserve order
        combined_df['original_order'] = range(len(combined_df))
        combined_df.to_csv('results/founder_insights.csv', index=False)


def combine_iterative_training_data():
    import os
    from pathlib import Path

    # Define the directory containing the iterative results
    results_dir = Path('iterative_train_data')
    
    # Initialize an empty list to store all dataframes
    all_dfs = []
    
    # Read all CSV files from the iterative_train_data directory in sorted order
    for file in sorted(results_dir.glob('*.csv')):
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all dataframes and save to a new CSV file
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Add an index column to preserve order
        combined_df['original_order'] = range(len(combined_df))
        output_file = 'iterative_train_data/combined_data.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Combined results saved to {output_file}")

def create_founder_insights_csv():
    founders_df = pd.read_csv('iterative_train_data/combined_data.csv')
    insights_df = pd.read_csv('results/founder_insights.csv')

def merge_and_validate_results():
    # Read both files
    founder_insights = pd.read_csv('founder_insights.csv')
    combined_results = pd.read_csv('founder_data.csv')
    
    # Use original_order for merging if it exists, otherwise create one
    if 'original_order' in founder_insights.columns and 'original_order' in combined_results.columns:
        founder_insights = founder_insights.sort_values('original_order')
        combined_results = combined_results.sort_values('original_order')
    else:
        founder_insights['original_order'] = range(len(founder_insights))
        combined_results['original_order'] = range(len(combined_results))
    
    # Validate that founder profiles match
    print("\nValidating founder profiles match:")
    mismatches = 0
    for idx in range(len(founder_insights)):
        founder_name1 = founder_insights.iloc[idx]['founder_name']
        founder_name2 = combined_results.iloc[idx]['founder_name']
        founder_success1 = founder_insights.iloc[idx]['success']
        founder_success2 = combined_results.iloc[idx]['success']
        
        if founder_name1 != founder_name2 or founder_success1 != founder_success2:
            print(f"Mismatch found at index {idx}")
            print(f"Founder insights profile: {founder_name1[:100]}...")
            print(f"Combined results profile: {founder_name2[:100]}...")
            mismatches += 1
            raise Exception("Founder profiles do not match")
    
    if mismatches == 0:
        print("All founder profiles match!")
    else:
        print(f"\nTotal mismatches found: {mismatches}")
        return None
    
    # Merge the dataframes based on original_order
    merged_df = pd.merge(
        founder_insights[['original_order', 'founder_name', 'success', 'insight']],
        combined_results.drop(['founder_name', 'success'], axis=1),  # Drop these columns from combined_results
        on='original_order',
        suffixes=('_founder', '_iterative')
    )
    
    # Save the merged dataframe
    output_file = 'founder_data_with_insights.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\nMerged results saved to {output_file}")
    
    return merged_df

def generate_cross_validation_folders():
    import os
    from itertools import combinations
    
    # Create cross_validation_data directory if it doesn't exist
    os.makedirs('cross_validation_data', exist_ok=True)
    
    # List of all folds
    folds = [0, 1, 2, 3]
    
    # Generate all possible combinations of 2 folds for training
    train_combinations = list(combinations(folds, 2))
    
    # For each training combination, generate validation/test splits
    for i, train_folds in enumerate(train_combinations):
        remaining_folds = [f for f in folds if f not in train_folds]
        
        # For each remaining fold, use it as validation and the other as test
        for j, val_fold in enumerate(remaining_folds):
            test_fold = [f for f in remaining_folds if f != val_fold][0]
            
            # Create folder for this combination
            folder_name = f'cross_validation_data/fold_{i*2 + j}'
            os.makedirs(folder_name, exist_ok=True)
            
            # Read and combine the training folds
            train_data = pd.concat([
                pd.read_csv(f'founder_data_with_insights_fold_{fold}.csv')
                for fold in train_folds
            ], ignore_index=True)
            
            # Read validation and test data
            val_data = pd.read_csv(f'founder_data_with_insights_fold_{val_fold}.csv')
            test_data = pd.read_csv(f'founder_data_with_insights_fold_{test_fold}.csv')
            
            # Save the data
            train_data.to_csv(f'{folder_name}/train_data.csv', index=False)
            val_data.to_csv(f'{folder_name}/validation_data.csv', index=False)
            test_data.to_csv(f'{folder_name}/test_data.csv', index=False)
            
            print(f"Created combination {i*2 + j}:")
            print(f"  Training folds: {train_folds}")
            print(f"  Validation fold: {val_fold}")
            print(f"  Test fold: {test_fold}")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Validation samples: {len(val_data)}")
            print(f"  Test samples: {len(test_data)}")
            print()


generate_cross_validation_folders()





