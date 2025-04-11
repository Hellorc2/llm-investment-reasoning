import os
import glob

def main():
    # Get all train batch files
    files = glob.glob('iterative_test_data/train_batch_*.csv')
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    # Create a mapping of old names to new names
    for i, old_file in enumerate(files, start=0):
        # Create new filename with zero-padded index
        new_file = os.path.join('iterative_test_data', f'test_batch_{i:03d}.csv')
        
        try:
            os.rename(old_file, new_file)
            print(f'Renamed {os.path.basename(old_file)} to {os.path.basename(new_file)}')
        except Exception as e:
            print(f'Error renaming {old_file}: {e}')

if __name__ == '__main__':
    main() 