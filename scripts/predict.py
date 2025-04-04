from G_generate_problog import generate_problog_program, generate_base_problog_program
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
from problog.program import PrologString
from problog import get_evaluatable


def predict_success_of_founder(row_number, founder_info, iteration_index, threshold_success, threshold_failure):
    # Convert pandas Series to dictionary
    founder_dict = founder_info.to_dict()
    
    # Use process-specific file
    import os
    from filelock import FileLock
    process_id = os.getpid()
    program_file = f'problog_program_{process_id}.pl'
    lock_file = f'problog_program_{process_id}.pl.lock'
    
    try:
        # Create a lock for this process's file
        lock = FileLock(lock_file)
        
        with lock:  # This ensures exclusive access to the file
            # Generate program in process-specific file
            generate_problog_program(iteration_index, founder_dict, program_file)
            
            # Read the generated program
            with open(program_file, 'r') as f:
                program = f.read()
            
            # Create and evaluate the model using the 'approximate' engine
            model = PrologString(program)
            result = get_evaluatable().create_from(model).evaluate(engine='approximate')
            
            print(row_number, "Problog result:", result)  # Debug print to see the actual format
            
            # Handle empty result case
            if not result:
                print(f"Warning: Empty result for row {row_number}")
                return f'{row_number},0.0,1.0,{founder_info["success"]}\n'
            
            # Convert result to dictionary and get success probability
            result_dict = {str(k): float(v) for k, v in result.items()}
            success_prob = result_dict.get('success', 0.0)
            failure_prob = result_dict.get('failure', 1.0)
            
            is_success = founder_info['success']
            
            return f'{row_number},{success_prob},{failure_prob},{is_success}\n'
            
    finally:
        # Clean up both the program file and its lock file
        try:
            if os.path.exists(program_file):
                os.remove(program_file)
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            print(f"Warning: Could not clean up files for process {process_id}: {str(e)}")




def predict_series(csv_file, iteration_index, threshold_success = 0.2, threshold_failure = 0.999):
    """Process rows serially instead of in parallel"""
    # Read the data using pandas
    data = pd.read_csv(csv_file)
    
    # Clear prediction.csv before starting predictions
    import shutil
    try:
        shutil.copyfile(f'prediction_{iteration_index}.csv', f'prediction_copy_{iteration_index}.csv')
    except FileNotFoundError:
        pass  # Ignore if original file doesn't exist yet
    
    with open(f'prediction_{iteration_index}.csv', 'w') as f:
        f.write('row_number,success_prob,failure_prob,is_success\n')
    
    # Track rows that need reprocessing
    rows_to_reprocess = []
    
    # Process rows serially
    total_rows = len(data)
    for idx, row in data.iterrows():
        try:
            result = predict_success_of_founder(idx, row, iteration_index, threshold_success, threshold_failure)
            
            # Parse the result string
            row_num, success_prob, failure_prob, is_success = result.strip().split(',')
            success_prob = float(success_prob)
            failure_prob = float(failure_prob)
            
            # If success_prob is 0 and failure_prob is 1, mark for reprocessing
            if success_prob == 0.0 and failure_prob == 1.0:
                rows_to_reprocess.append(int(row_num))
            
            # Write result immediately
            with open(f'prediction_{iteration_index}.csv', 'a') as f:
                f.write(result)
            
            print(f"Processed row {idx} out of {total_rows}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            rows_to_reprocess.append(int(idx))
    
    # Reprocess failed rows if any
    if rows_to_reprocess:
        print(f"\nReprocessing {len(rows_to_reprocess)} rows with zero success probability...")
        
        # Process failed rows serially
        reprocess_results = []
        for idx in rows_to_reprocess:
            try:
                result = predict_success_of_founder(idx, data.iloc[idx], iteration_index, threshold_success, threshold_failure)
                reprocess_results.append(result)
                print(f"Reprocessed row {idx}")
            except Exception as e:
                print(f"Error reprocessing row {idx}: {str(e)}")
        
        # Update the results in the CSV file
        with open(f'prediction_{iteration_index}.csv', 'r') as f:
            lines = f.readlines()
        
        # Create a mapping of row numbers to new results
        result_map = {}
        for result in reprocess_results:
            row_num, _, _, _ = result.strip().split(',')
            result_map[int(row_num)] = result
        
        # Update the lines with new results
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
            row_num = int(line.split(',')[0])
            if row_num in result_map:
                lines[i] = result_map[row_num]
        
        # Write updated results back to file
        with open(f'prediction_{iteration_index}.csv', 'w') as f:
            f.writelines(lines)
        
        print(f"Reprocessing complete. Updated {len(rows_to_reprocess)} rows.")
    
    print("finished")

def predict(csv_file, iteration_index, threshold_success = 0.2, threshold_failure = 0.999):
    # Read the data using pandas
    data = pd.read_csv(csv_file)
    
    # Clear prediction.csv before starting predictions
    import shutil
    try:
        shutil.copyfile(f'prediction_{iteration_index}.csv', f'prediction_copy_{iteration_index}.csv')
    except FileNotFoundError:
        pass  # Ignore if original file doesn't exist yet
    
    generate_base_problog_program(iteration_index)

    # Create list of (index, row) tuples
    rows_to_process = list(data.iterrows())
    print(f"Total rows to process: {len(rows_to_process)}")
    
    # Create a pool of workers
    num_processes = min(cpu_count(), len(rows_to_process))  # Don't use more processes than rows
    print(f"Using {num_processes} processes")
    pool = Pool(processes=num_processes)
    
    try:
        # Process all rows in parallel
        results = pool.starmap(predict_success_of_founder, 
                             [(idx, row, iteration_index, threshold_success, threshold_failure) 
                              for idx, row in rows_to_process])
        
        # Check results and track rows that need reprocessing
        rows_to_reprocess = []
        for result in results:
            # Parse the result string
            row_num, success_prob, failure_prob, is_success = result.strip().split(',')
            success_prob = float(success_prob)
            failure_prob = float(failure_prob)
            
            # If success_prob is 0 and failure_prob is 1, mark for reprocessing
            if success_prob == 0.0 and failure_prob == 1.0:
                rows_to_reprocess.append(int(row_num))
        
        # Reprocess failed rows if any
        if rows_to_reprocess:
            print(f"\nReprocessing {len(rows_to_reprocess)} rows with zero success probability...")
            
            # Process failed rows serially
            reprocess_results = []
            for idx in rows_to_reprocess:
                result = predict_success_of_founder(idx, data.iloc[idx], iteration_index, threshold_success, threshold_failure)
                reprocess_results.append(result)
                print(f"Reprocessed row {idx}")
            
            # Create a mapping of row numbers to new results
            result_map = {}
            for result in reprocess_results:
                row_num, _, _, _ = result.strip().split(',')
                result_map[int(row_num)] = result
            
            # Update the results with reprocessed values
            for i, result in enumerate(results):
                row_num = int(result.strip().split(',')[0])
                if row_num in result_map:
                    results[i] = result_map[row_num]
        
        # Write all results at once
        print(f"\nWriting {len(results)} results to prediction_{iteration_index}.csv")
        with open(f'prediction_{iteration_index}.csv', 'w') as f:
            f.write('row_number,success_prob,failure_prob,is_success\n')
            f.writelines(results)
        
        # Verify the file was written correctly
        try:
            df = pd.read_csv(f'prediction_{iteration_index}.csv')
            print(f"Verified file contains {len(df)} rows")
        except Exception as e:
            print(f"Error verifying file: {str(e)}")
        
    finally:
        # Always close the pool
        pool.close()
        pool.join()
    
    print("finished")

def manual_prediction_analysis(iteration_index, threshold_success = 0.3, threshold_failure = 1):
    predictions_df = pd.read_csv(f'prediction_{iteration_index}.csv')
    # Add a predictions column based on thresholds
    predictions_df['prediction'] = ((predictions_df['success_prob'] > threshold_success) & 
                                  (predictions_df['failure_prob'] < threshold_failure)).astype(int)
    
    # Calculate accuracy metrics
    total_predictions = len(predictions_df)
    correct_predictions = sum(predictions_df['prediction'] == predictions_df['is_success'])
    accuracy = correct_predictions / total_predictions

    # Calculate precision
    true_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 1))
    false_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 0))
    false_negatives = sum((predictions_df['prediction'] == 0) & (predictions_df['is_success'] == 1))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"\nPrediction Summary for success threshold {threshold_success} and failure threshold {threshold_failure}:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Number of true positives: {true_positives}")
    print(f"Number of false positives: {false_positives}")
    """
    with open(f'prediction_summary_{iteration_index}.txt', 'a') as f:
        f.write(f"Prediction Summary for Iteration {iteration_index}\n")
        f.write(f"Success Threshold: {threshold_success}\n")
        f.write(f"Failure Threshold: {threshold_failure}\n")
        f.write(f"Total predictions: {total_predictions}\n")
        f.write(f"Correct predictions: {correct_predictions}\n") 
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.2%}\n")
        f.write(f"Number of true positives: {true_positives}\n")
        f.write(f"Number of false positives: {false_positives}\n")
    """
    
    return precision,accuracy, recall

def plot_precision_analysis(iterations, failure_thresholds = [0.999, 0.9999, 0.99999]):
    import numpy as np
    import matplotlib.pyplot as plt

    # Store results for plotting
    results = []
    for iteration_index in iterations:
        for j in failure_thresholds:
            for i in np.arange(0.05, 0.95, 0.05):
                precision,accuracy,recall = manual_prediction_analysis(iteration_index, i, j)
                results.append({
                    'iteration': iteration_index,
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall
                })

    # Convert results to DataFrame for easier plotting
    results_df = pd.DataFrame(results)

    # Create three separate plots, one for each failure threshold
    for j in failure_thresholds:
        plt.figure(figsize=(12, 6))
        
        # Filter data for this failure threshold
        threshold_data = results_df[results_df['threshold_failure'] == j]
        
        # Plot lines for each iteration
        for iteration in threshold_data['iteration'].unique():
            iteration_data = threshold_data[threshold_data['iteration'] == iteration]
            plt.plot(iteration_data['threshold_success'], iteration_data['precision'], 
                    label=f'Iteration {iteration}', marker='o')
        
        plt.xlabel('Success Threshold')
        plt.ylabel('Precision')
        plt.title(f'Precision vs Success Threshold (Failure Threshold = {j})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'precision_analysis_failure_threshold_{j}.png')
        plt.close()

    for j in failure_thresholds:
        plt.figure(figsize=(12, 6))
        
        # Filter data for this failure threshold
        threshold_data = results_df[results_df['threshold_failure'] == j]
        
        # Plot lines for each iteration
        for iteration in threshold_data['iteration'].unique():
            iteration_data = threshold_data[threshold_data['iteration'] == iteration]
            plt.plot(iteration_data['threshold_success'], iteration_data['accuracy'], 
                    label=f'Iteration {iteration}', marker='o')
        
        plt.xlabel('Success Threshold')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Success Threshold (Failure Threshold = {j})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'accuracy_analysis_failure_threshold_{j}.png')
        plt.close()


    # Print tables for each iteration and failure threshold
    print("\nResults Tables:")
    print("=" * 80)
    
    for iteration in iterations:
        print(f"\nIteration {iteration}:")
        print("-" * 80)
        print(f"{'Failure Threshold':<20} {'Success Threshold':<20} {'Precision':<10} {'Recall':<10} {'True Positives':<15} {'False Positives':<15}")
        print("-" * 80)
        
        for failure_threshold in failure_thresholds:
            for success_threshold in np.arange(0.05, 0.95, 0.05):
                # Get predictions for this combination
                predictions_df = pd.read_csv(f'prediction_{iteration}.csv')
                predictions_df['prediction'] = ((predictions_df['success_prob'] > success_threshold) & 
                                              (predictions_df['failure_prob'] < failure_threshold)).astype(int)
                
                true_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 1))
                false_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 0))
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                false_negatives = sum((predictions_df['prediction'] == 0) & (predictions_df['is_success'] == 1))
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                print(f"{failure_threshold:<20} {success_threshold:<20.2f} {precision:<10.3f} {recall:<10.3f} {true_positives:<15d} {false_positives:<15d}")
        print("-" * 80)
    print("=" * 80)



def average_prediction(iterations):
    """
    Calculate average success and failure probabilities across iterations.
    
    Args:
        iterations (list): List of iteration numbers to average over
        
    Returns:
        pd.DataFrame: DataFrame with averaged probabilities
    """
    # Read first prediction file to get number of rows
    base_df = pd.read_csv(f'prediction_{iterations[0]}.csv')
    num_rows = len(base_df)
    
    # Initialize arrays to store sums
    success_sums = np.zeros(num_rows)
    failure_sums = np.zeros(num_rows)
    
    # Sum up probabilities across iterations
    for iteration in iterations:
        df = pd.read_csv(f'prediction_{iteration}.csv')
        success_sums += df['success_prob'].values
        failure_sums += df['failure_prob'].values
    
    # Calculate averages
    avg_success = success_sums / len(iterations)
    avg_failure = failure_sums / len(iterations)
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'success_prob': avg_success,
        'failure_prob': avg_failure,
        'is_success': base_df['is_success']
    })
    
    # Save to CSV file
    result_df.to_csv('prediction_-1.csv', index=False)
    
    return result_df

def max_prediction(iterations):
    base_df = pd.read_csv(f'prediction_{iterations[0]}.csv')
    num_rows = len(base_df)
    
    # Initialize arrays to store max success and min failure
    max_success = np.zeros(num_rows)
    min_failure = np.ones(num_rows)
    
    # Find max success and min failure across iterations
    for iteration in iterations:
        df = pd.read_csv(f'prediction_{iteration}.csv')
        max_success = np.maximum(max_success, df['success_prob'].values)
        min_failure = np.minimum(min_failure, df['failure_prob'].values)

    
    # Create output dataframe
    result_df = pd.DataFrame({
        'success_prob': max_success,
        'failure_prob': min_failure,
        'is_success': base_df['is_success']
    })
    
    # Save to CSV file
    result_df.to_csv('prediction_-1.csv', index=False)
    
    return result_df

if __name__ == "__main__":
    predict('test_data_reduced.csv', 19)