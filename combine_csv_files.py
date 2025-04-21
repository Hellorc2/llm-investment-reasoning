import pandas as pd
import glob
import os

# Get all CSV files except the discarded one
csv_files = glob.glob('iterative_train_data/train_batch_*.csv')
csv_files = [f for f in csv_files if 'discarded' not in f]

# Sort files to maintain order
csv_files.sort()

# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV file and append to the list
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = 'iterative_train_data/combined_train_data.csv'
combined_df.to_csv(output_file, index=False)

print(f"Successfully combined {len(csv_files)} files into {output_file}")
print(f"Total number of rows in combined file: {len(combined_df)}") 