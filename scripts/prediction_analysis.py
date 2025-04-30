import pandas as pd
import numpy as np
from datetime import datetime

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
    f_score_half = (5/4) / ((1/precision) + (1/4)*(1/recall)) if (precision + recall) > 0 else 0
    f_score_quarter = (1+1/16) / ((1/precision) + (1/16)*(1/recall)) if (precision + recall) > 0 else 0
    
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
    
    return precision, accuracy, recall, f_score_half, f_score_quarter, true_positives

def get_best_models(iterations, success_thresholds = np.arange(0.01, 1, 0.01), 
                    failure_thresholds = np.concatenate([np.arange(0.8, 1, 0.02), np.arange(0.99, 1, 0.001), 
                                                         np.arange(0.999, 0.9999, 0.0001), np.arange(0.9999, 1, 0.00001), np.arange(0.99999, 1, 0.000001)]), 
                    iterative = False, num_top_results = 1, f_score_parameter = 0.5):
    results = {}
    best_success_thresholds = {}
    best_failure_thresholds = {}
    
    for iteration_index in iterations:
        results_iteration = []
        for i in success_thresholds:
            for j in failure_thresholds:
                precision,accuracy,recall, f_score_half, f_score_quarter, true_positives = prediction_analysis(iteration_index, i, j, iterative)
                results_iteration.append({
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f_score_half': f_score_half,
                    'f_score_quarter': f_score_quarter,
                    'true_positives': true_positives
                })
        results[iteration_index] = results_iteration

    best_models_str = ""
    # Convert results to DataFrame for easier plotting
    for iteration_index in iterations:
        results_iteration_df = pd.DataFrame(results[iteration_index])

        # Create the output string
        output = f"Iteration {iteration_index} - Top {num_top_results} configurations:\n"
        output += "-" * 50 + "\n"
        
        if f_score_parameter == 0.5:
            top_results = results_iteration_df.sort_values('f_score_half', ascending=False).head(num_top_results)
        elif f_score_parameter == 0.25:
            top_results = results_iteration_df.sort_values('f_score_quarter', ascending=False).head(num_top_results)
        
        # Store the best thresholds for this iteration
        best_result = top_results.iloc[0]
        best_success_thresholds[iteration_index] = best_result.threshold_success
        best_failure_thresholds[iteration_index] = best_result.threshold_failure
        
        for i, result in enumerate(top_results.itertuples(), 1):
            output += f"#{i}:\n"
            output += f"F-{f_score_parameter}score: {result.f_score_half:.3f}\n" if f_score_parameter == 0.5 else f"F-{f_score_parameter}score: {result.f_score_quarter:.3f}\n"
            output += f"Success Threshold: {result.threshold_success:.2f}\n"
            output += f"Failure Threshold: {result.threshold_failure}\n"
            output += f"Precision: {result.precision:.3f}\n"
            output += f"Recall: {result.recall:.3f}\n"
            output += "\n"
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_log/best_models_iteration_{iteration_index:03d}_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(output)
        
        # Also print to console
        print(output)
        best_models_str += output

    return best_models_str, best_success_thresholds, best_failure_thresholds





def plot_precision_analysis(iterations, failure_thresholds = [0.999, 0.9999, 0.99999], iterative = False):
    import matplotlib.pyplot as plt
    # Store results for plotting
    results = []
    for iteration_index in iterations:
        for j in failure_thresholds:
            for i in np.arange(0.05, 0.95, 0.05):
                precision,accuracy,recall, f_score_half, f_score_quarter, true_positives = prediction_analysis(iteration_index, i, j, iterative)
                results.append({
                    'iteration': iteration_index,
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f_score_half': f_score_half,
                    'f_score_quarter': f_score_quarter,
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

                precision, accuracy, recall, f_score_half, f_score_quarter, true_positives = prediction_analysis(iteration, success_threshold, failure_threshold, iterative)
                
                print(f"{failure_threshold:<20} {success_threshold:<20.2f} {precision:<10.3f} {recall:<10.3f} {true_positives:<15d}")
        print("-" * 80)
    print("=" * 80)

def get_best_model_clusters(iterations, success_thresholds = np.arange(0.01, 1, 0.01),
                      failure_thresholds = np.concatenate([np.arange(0.8, 1, 0.02), np.arange(0.99, 1, 0.001), 
                                                         np.arange(0.999, 0.9999, 0.0001), np.arange(0.9999, 1, 0.00001), np.arange(0.99999, 1, 0.000001)]), 
                      iterative = False, num_top_results = 1, f_score_parameter = 0.5):
    results = {}
    best_success_thresholds = {}
    best_failure_thresholds = {}
    
    for iteration_index in iterations:
        results_iteration = []
        for i in success_thresholds:
            for j in failure_thresholds:
                precision, accuracy, recall, f_score_half, f_score_quarter, true_positives = prediction_analysis(iteration_index, i, j, iterative)
                results_iteration.append({
                    'threshold_success': i,
                    'threshold_failure': j,
                    'precision': precision,
                    'accuracy': accuracy,
                    'recall': recall,
                    'f_score_half': f_score_half,
                    'f_score_quarter': f_score_quarter,
                    'true_positives': true_positives
                })
        results[iteration_index] = results_iteration

    best_models_str = ""
    # Convert results to DataFrame for easier analysis
    for iteration_index in iterations:
        results_iteration_df = pd.DataFrame(results[iteration_index])
        
        # Create grids for each metric
        success_grid = np.unique(results_iteration_df['threshold_success'])
        failure_grid = np.unique(results_iteration_df['threshold_failure'])
        f_score_grid = np.zeros((len(success_grid), len(failure_grid)))
        precision_grid = np.zeros((len(success_grid), len(failure_grid)))
        recall_grid = np.zeros((len(success_grid), len(failure_grid)))
        
        # Fill the grids with metrics
        for idx, row in results_iteration_df.iterrows():
            i = np.where(success_grid == row['threshold_success'])[0][0]
            j = np.where(failure_grid == row['threshold_failure'])[0][0]
            if f_score_parameter == 0.5:
                f_score_grid[i, j] = row['f_score_half']
            elif f_score_parameter == 0.25:
                f_score_grid[i, j] = row['f_score_quarter']
            precision_grid[i, j] = row['precision']
            recall_grid[i, j] = row['recall']
        
        # Calculate local averages
        local_averages = []
        for i in range(len(success_grid)):
            for j in range(len(failure_grid)):
                # Define the 11x11 neighborhood (-5 to +5)
                i_start = max(0, i-5)
                i_end = min(len(success_grid), i+6)
                j_start = max(0, j-5)
                j_end = min(len(failure_grid), j+6)
                
                # Calculate averages in the neighborhood
                f_score_neighborhood = f_score_grid[i_start:i_end, j_start:j_end]
                precision_neighborhood = precision_grid[i_start:i_end, j_start:j_end]
                recall_neighborhood = recall_grid[i_start:i_end, j_start:j_end]
                
                local_avg_f_score = np.mean(f_score_neighborhood)
                local_avg_precision = np.mean(precision_neighborhood)
                local_avg_recall = np.mean(recall_neighborhood)
                
                local_averages.append({
                    'threshold_success': success_grid[i],
                    'threshold_failure': failure_grid[j],
                    'local_avg_f_score': local_avg_f_score,
                    'local_avg_precision': local_avg_precision,
                    'local_avg_recall': local_avg_recall,
                    'original_f_score': f_score_grid[i, j],
                    'original_precision': precision_grid[i, j],
                    'original_recall': recall_grid[i, j]
                })
        
        # Convert to DataFrame and sort by local average F-score
        local_avg_df = pd.DataFrame(local_averages)
        local_avg_df = local_avg_df.sort_values('local_avg_f_score', ascending=False)
        
        # Store the best thresholds for this iteration
        best_result = local_avg_df.iloc[0]
        best_success_thresholds[iteration_index] = best_result.threshold_success
        best_failure_thresholds[iteration_index] = best_result.threshold_failure
        
        # Create the output string
        output = f"Iteration {iteration_index} - Top {num_top_results} configurations by local average F-score {f_score_parameter}:\n"
        output += "-" * 50 + "\n"
        
        for i, result in enumerate(local_avg_df.head(num_top_results).itertuples(), 1):
            output += f"#{i}:\n"
            output += f"Local Average F-{f_score_parameter}score: {result.local_avg_f_score:.3f}\n"
            output += f"Local Average Precision: {result.local_avg_precision:.3f}\n"
            output += f"Local Average Recall: {result.local_avg_recall:.3f}\n"
            output += f"Original F-{f_score_parameter}score: {result.original_f_score:.3f}\n"
            output += f"Original Precision: {result.original_precision:.3f}\n"
            output += f"Original Recall: {result.original_recall:.3f}\n"
            output += f"Success Threshold: {result.threshold_success:.2f}\n"
            output += f"Failure Threshold: {result.threshold_failure}\n"
            output += "\n"
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_log/best_model_clusters_iteration_{iteration_index:03d}_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(output)
        
        # Also print to console
        print(output)
        best_models_str += output

    return best_models_str, best_success_thresholds, best_failure_thresholds

def get_group_analysis(iterations, success_thresholds, failure_thresholds, majority_threshold = 1,iterative = False):
    predictions_group_df = pd.DataFrame()
    for iteration_index in iterations:
        if not iterative:
            file_name = f'predictions/prediction_{iteration_index}.csv'
        else:
            file_name = f'predictions_iterative/prediction_iterative_{iteration_index}.csv'
        success_threshold = success_thresholds[iteration_index]
        failure_threshold = failure_thresholds[iteration_index]
        predictions_df = pd.read_csv(file_name)
        # Add a predictions column based on thresholds
        predictions_group_df[f'prediction_{iteration_index}'] = ((predictions_df['success_prob'] > success_threshold) & 
                                    (predictions_df['failure_prob'] < failure_threshold)).astype(int)
        
    success_column = pd.read_csv(file_name)['is_success']
    predictions_group_df['is_success'] = success_column

    # Calculate majority predictions
    prediction_columns = [col for col in predictions_group_df.columns if col.startswith('prediction_')]
    predictions_group_df['majority_prediction'] = (
        predictions_group_df[prediction_columns].sum(axis=1) >= 
        (majority_threshold)
    ).astype(int)

    # Calculate accuracy metrics
    total_predictions = len(predictions_group_df)
    correct_predictions = sum(predictions_group_df['majority_prediction'] == predictions_group_df['is_success'])
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate precision
    true_positives = sum((predictions_group_df['majority_prediction'] == 1) & (predictions_group_df['is_success'] == 1))
    false_positives = sum((predictions_group_df['majority_prediction'] == 1) & (predictions_group_df['is_success'] == 0))
    false_negatives = sum((predictions_group_df['majority_prediction'] == 0) & (predictions_group_df['is_success'] == 1))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_score_half = (5/4) / ((1/precision) + (1/4)*(1/recall)) if (precision + recall) > 0 else 0
    f_score_quarter = (1+1/16) / ((1/precision) + (1/16)*(1/recall)) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f_score_half, f_score_quarter, true_positives



if __name__ == "__main__":
    best_models_str, best_success_thresholds, best_failure_thresholds = get_best_models(iterations = [0,1,2,3,4,5,6,7,8,9], num_top_results = 1, f_score_parameter = 0.25)
