import os
import glob
import pandas as pd

def main():
    # Get all train batch files
    files = glob.glob('iterative_test_data/train_batch_*.csv')
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    # Process files in groups of 5
    for group_idx, i in enumerate(range(0, len(files), 5)):
        # Get the next 5 files (or remaining files if less than 5)
        group_files = files[i:i+5]
        
        # Combine the files
        combined_df = pd.concat([pd.read_csv(f) for f in group_files], ignore_index=True)
        
        # Create new filename with zero-padded index
        new_file = os.path.join('iterative_test_data', f'test_batch_{group_idx:03d}.csv')
        
        try:
            # Save the combined file
            combined_df.to_csv(new_file, index=False)
            print(f'Created combined file {os.path.basename(new_file)} from:')
            for f in group_files:
                print(f'  - {os.path.basename(f)}')
                # Remove the original file after successful combination
                os.remove(f)
        except Exception as e:
            print(f'Error processing files {group_files}: {e}')

if __name__ == '__main__':
    main() 