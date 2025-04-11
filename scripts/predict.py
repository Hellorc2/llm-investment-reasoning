from generate_problog import generate_problog_program, generate_base_problog_program
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
from problog.program import PrologString
from problog import get_evaluatable
import numpy as np
import matplotlib.pyplot as plt


def predict_success_of_founder(row_number, founder_info, iteration_index):
    # Convert pandas Series to dictionary
    founder_dict = founder_info.to_dict()
    
    # Use process-specific file
    import os
    from filelock import FileLock
    process_id = os.getpid()
    program_file = f'problog_programs/problog_program_{process_id}.pl'
    lock_file = f'problog_programs/problog_program_{process_id}.pl.lock'
    
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



def predict(csv_file, iteration_index, iterative = False):
    # Read the data using pandas
    data = pd.read_csv(csv_file)
    
    if not iterative:
    # Clear prediction.csv before starting predictions
        import shutil
        try:
            shutil.copyfile(f'predictions/prediction_{iteration_index}.csv', f'predictions/prediction_copy_{iteration_index}.csv')
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
                             [(idx, row, iteration_index) 
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
                result = predict_success_of_founder(idx, data.iloc[idx], iteration_index)
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
        if not iterative:
            # Write all results at once
            print(f"\nWriting {len(results)} results to prediction_{iteration_index}.csv")
            with open(f'predictions/prediction_{iteration_index}.csv', 'w') as f:
                f.write('row_number,success_prob,failure_prob,is_success\n')
                f.writelines(results)
        else:
            print(f"\nWriting {len(results)} results to prediction_iterative_{iteration_index}.csv")
            with open(f'predictions_iterative/prediction_iterative_{iteration_index}.csv', 'w') as f:
                f.write('row_number,success_prob,failure_prob,is_success\n')
                f.writelines(results)

        
        # Verify the file was written correctly
        try:
            df = pd.read_csv(f'predictions/prediction_{iteration_index}.csv')
            print(f"Verified file contains {len(df)} rows")
        except Exception as e:
            print(f"Error verifying file: {str(e)}")
        
    finally:
        # Always close the pool
        pool.close()
        pool.join()
    
    print("finished")

def prediction_analysis(iteration_index, threshold_success = 0.3, threshold_failure = 1, iterative = False):
    if not iterative:
        predictions_df = pd.read_csv(f'predictions/prediction_{iteration_index}.csv')
    else:
        predictions_df = pd.read_csv(f'predictions_iterative/prediction_iterative_{iteration_index}.csv')
    # Add a predictions column based on thresholds
    predictions_df['prediction'] = ((predictions_df['success_prob'] > threshold_success) & 
                                  (predictions_df['failure_prob'] < threshold_failure)).astype(int)
    
    # Calculate accuracy metrics
    total_predictions = len(predictions_df)
    correct_predictions = sum(predictions_df['prediction'] == predictions_df['is_success'])
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate precision
    true_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 1))
    false_positives = sum((predictions_df['prediction'] == 1) & (predictions_df['is_success'] == 0))
    false_negatives = sum((predictions_df['prediction'] == 0) & (predictions_df['is_success'] == 1))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_score = (5/4) / ((1/precision) + (1/4)*(1/recall)) if (precision + recall) > 0 else 0
    
    """
    print(f"\nPrediction Summary for success threshold {threshold_success} and failure threshold {threshold_failure}:")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"Number of true positives: {true_positives}")
    print(f"Number of false positives: {false_positives}")
"""

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
    
    return precision, accuracy, recall, f_score, true_positives

def get_best_models(iterations, success_thresholds = np.arange(0.05, 0.95, 0.05), failure_thresholds = np.append(np.arange(0.8, 1, 0.02),[0.999,0.9999]), iterative = False):
        # Store results for plotting
    results = []
    for iteration_index in iterations:
        for i in success_thresholds:
            for j in failure_thresholds:
                precision,accuracy,recall, f_score, true_positives = prediction_analysis(iteration_index, i, j, iterative)
                results.append({
                    'iteration': iteration_index,
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f_score': f_score,
                    'true_positives': true_positives
                })

    # Convert results to DataFrame for easier plotting
    results_df = pd.DataFrame(results)

    best_models_str = ""
        # For each iteration, save top 3 results to a file
    for iteration in iterations:
        # Create the output string
        output = f"Iteration {iteration} - Top 3 configurations:\n"
        output += "-" * 50 + "\n"
        
        top_3_results = results_df[results_df['iteration'] == iteration].sort_values('f_score', ascending=False).head(3)
        for i, result in enumerate(top_3_results.itertuples(), 1):
            output += f"#{i}:\n"
            output += f"F-score: {result.f_score:.3f}\n"
            output += f"Success Threshold: {result.threshold_success:.2f}\n"
            output += f"Failure Threshold: {result.threshold_failure}\n"
            output += f"Precision: {result.precision:.3f}\n"
            output += f"Recall: {result.recall:.3f}\n"
            output += "\n"
        
        # Save to file
        filename = f"predictions_iterative/top_3_results_iteration_{iteration:03d}.txt"
        with open(filename, 'w') as f:
            f.write(output)
        
        # Also print to console
        print(output)
        best_models_str += output

    return best_models_str

def plot_precision_analysis(iterations, failure_thresholds = [0.999, 0.9999, 0.99999], iterative = False):
    import matplotlib.pyplot as plt
    # Store results for plotting
    results = []
    for iteration_index in iterations:
        for j in failure_thresholds:
            for i in np.arange(0.05, 0.95, 0.05):
                precision,accuracy,recall, f_score, true_positives = prediction_analysis(iteration_index, i, j)
                results.append({
                    'iteration': iteration_index,
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f_score': f_score,
                    'true_positives': true_positives
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


    # Print tables for each iteration and failure threshold
    print("\nResults Tables:")
    print("=" * 80)
    
    for iteration in iterations:
        print(f"\nIteration {iteration}:")
        print("-" * 80)
        print(f"{'Failure Threshold':<20} {'Success Threshold':<20} {'Precision':<10} {'Recall':<10} {'True Positives':<15}")
        print("-" * 80)
        
        for failure_threshold in failure_thresholds:
            for success_threshold in np.arange(0.05, 0.95, 0.05):

                precision, accuracy, recall, f_score, true_positives = prediction_analysis(iteration, success_threshold, failure_threshold, iterative)
                
                print(f"{failure_threshold:<20} {success_threshold:<20.2f} {precision:<10.3f} {recall:<10.3f} {true_positives:<15d}")
        print("-" * 80)
    print("=" * 80)

if __name__ == "__main__":
    predict(csv_file = 'test_data_reduced.csv', iteration_index = 3, iterative = True)