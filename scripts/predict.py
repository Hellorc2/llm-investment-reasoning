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
            result = predict_success_of_founder(idx, row, iteration_index)
            
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
                result = predict_success_of_founder(idx, data.iloc[idx], iteration_index)
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
    results = []
    
    try:
        # Process all rows in parallel
        results = pool.starmap(predict_success_of_founder, 
                             [(idx, row, iteration_index) 
                              for idx, row in rows_to_process])
        
        """
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
        """
                
    finally:
        # Always close the pool
        pool.close()
        pool.join()
    
    # Reprocess failed rows if any
    """
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
    """

    # Write results after all processing is complete
    output_file = f'predictions_iterative/prediction_iterative_{iteration_index}.csv' if iterative else f'predictions/prediction_{iteration_index}.csv'
    print(f"\nWriting {len(results)} results to {output_file}")
    with open(output_file, 'w') as f:
        f.write('row_number,success_prob,failure_prob,is_success\n')
        f.writelines(results)
    
    # Write to prediction log with timestamp
    import datetime
    import os
    
    # Create prediction_log directory if it doesn't exist
    os.makedirs('prediction_log', exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    log_filename = f'prediction_log/{csv_file}_iter{iteration_index}_{iterative}_{timestamp}.csv'
    
    # Write to log file
    print(f"\nWriting {len(results)} results to log file: {log_filename}")
    with open(log_filename, 'w') as f:
        f.write('row_number,success_prob,failure_prob,is_success\n')
        f.writelines(results)
    
    print("finished")


if __name__ == "__main__":
    for i in range(0,9):
        predict(csv_file = 'test_data_validation.csv', iteration_index = i, iterative = False)