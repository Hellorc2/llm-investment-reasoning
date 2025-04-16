import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm  # For progress bar
import argparse


def arm_success(founders_data_sample,feature_combination, min_sample, random_sample_size = 1):
    # This function performs association rule mining to find patterns in the data that correlate with success.
    # The main goals are to find frequent feature combinations that appear in successful founders and to generate association rules.
    # This is useful for analyzing which combinations of characteristics are most strongly associated with founder success.


    # Calculate random success probability
    real_world_prob = 1.9
    # Set the real-world probability of success (1.9%) to adjust the baseline calculation.
    # This is used to normalize the probability of success in the sample to reflect real-world expectations.
    random_success_prob = (founders_data_sample['success'] == 1).mean() * 100

    # Calculate the random probability of success based on the percentage of successful founders in the dataset.
    # This serves as the baseline probability to compare against combinations found through association rule mining.
    adjusted_random_prob = real_world_prob
    print(f"Random probability of success (baseline): {random_success_prob:.2f}%")


    # Filter and encode relevant columns
    categorical_data = founders_data_sample.select_dtypes(include=['int64', 'bool', 'object']).copy()
    # Select only categorical columns for encoding and analysis. This ensures that numerical features are not mistakenly included.
    

    categorical_data_encoded = pd.get_dummies(categorical_data.drop(columns=['success']), columns=categorical_data.columns.drop('success')).astype(bool)
    # Convert categorical columns into binary format using one-hot encoding.
    # The success column is excluded because it's the target variable, not an input feature.

    # Filter out negative indicators (e.g., *_0 or *_False) to focus on positive features only
    positive_columns = [col for col in categorical_data_encoded.columns if not col.endswith('_0') and not col.endswith('_False') and not col.endswith('_nope')]
    # Filter out negative indicators (e.g., *_0, *_False, *_nope) to focus only on positive traits.
    # This ensures that only attributes indicative of positive experiences or characteristics are analyzed.
    categorical_data_encoded = categorical_data_encoded[positive_columns]


    # Use Apriori to find frequent itemsets with higher minimum support to reduce load
    frequent_itemsets = apriori(categorical_data_encoded, min_support=(min_sample / len(categorical_data_encoded)), use_colnames=True, verbose=1, max_len=feature_combination)
    # Use the Apriori algorithm to identify frequent feature combinations that meet the minimum support threshold.
    # The feature_combination parameter determines the maximum size of feature combinations (e.g., pairwise).
    if frequent_itemsets.empty:
        print("No frequent itemsets were generated. Please adjust the min_support value.")
        return

    # Filter itemsets to only include those with exactly the specified number of features
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == feature_combination)]
    # Filter frequent itemsets to include only those with exactly the specified number of features.
    # This ensures that the output matches the desired feature combination size (e.g., pairwise).

    # Calculate success probability for each itemset
    import math
    success_probabilities = []
    likelihood_of_success = []
    sample_counts = []
    standard_deviations = []
    confidence_intervals = []
    for _, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        # Filter the original sample to include rows that have all items in the itemset
        mask = pd.Series(True, index=founders_data_sample.index)
        # Create a boolean mask to filter the dataset for rows that match all features in the itemset.
        for item in itemset:
            if item in categorical_data_encoded.columns:
                mask &= categorical_data_encoded[item]
                filtered_data = founders_data_sample[mask]
        # Calculate success probability for the filtered data
        success_count = filtered_data['success'].sum()
        success_probability = (success_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
        # Calculate the success probability for the filtered data subset.
        # This measures the percentage of founders with the itemset features that were successful.
        likelihood = (success_probability / random_success_prob) if random_success_prob > 0 else 0
        likelihood_of_success.append(likelihood)
        sample_counts.append(len(filtered_data))
        # Calculate standard deviation based on binomial distribution
        p = success_probability / 100
        n = len(filtered_data)
        standard_deviation = math.sqrt(p * (1 - p) / n) * 100 if n > 0 else 0
        # Calculate 95% confidence interval
        Z = 1.96  # Z-score for 95% confidence level
        margin_of_error = Z * standard_deviation
        confidence_lower = max(0, success_probability - margin_of_error)
        confidence_upper = min(100, success_probability + margin_of_error)
        confidence_intervals.append((confidence_lower, confidence_upper))
        success_probabilities.append(success_probability)
        standard_deviations.append(standard_deviation)

    frequent_itemsets['success_probability'] = success_probabilities
    frequent_itemsets['sample_count'] = sample_counts
    frequent_itemsets['likelihood_of_success'] = likelihood_of_success
    frequent_itemsets['real_world_prob'] = frequent_itemsets['success_probability'] * (real_world_prob / random_success_prob)
    frequent_itemsets['confidence_interval_95'] = [(round(conf[0] * (real_world_prob / random_success_prob), 2), round(conf[1] * (real_world_prob / random_success_prob), 2)) for conf in confidence_intervals]
    # Add calculated columns to the frequent itemsets DataFrame, including:
    # - success_probability: Percentage of success for founders with the itemset features.
    # - sample_count: Number of samples matching the itemset.
    # - likelihood_of_success: Comparison of success probability to the random baseline.
    # - real_world_prob: Normalized success probability to reflect real-world distribution.
    # - confidence_interval_95: 95% confidence interval for success probability.

    # Display the number of frequent itemsets identified
    print(f"Number of frequent itemsets identified: {len(frequent_itemsets)}")

    # Display the top 10 frequent itemsets based on success probability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(f"Top Frequent Itemsets by Success Probability (including sample count, likelihood of success, and 95% confidence interval):")
    if not frequent_itemsets.empty:
        top_itemsets = frequent_itemsets[['itemsets', 'success_probability', 'sample_count', 'likelihood_of_success', 'real_world_prob', 'confidence_interval_95']].sort_values(by='success_probability', ascending=False).head(5)
        filtered_top_itemsets = top_itemsets[top_itemsets['sample_count'] >= 10]
        print(top_itemsets)
        random_selection = filtered_top_itemsets.sample(n=random_sample_size)
        rule_str = random_selection[['itemsets','success_probability']]
        print(rule_str)
    
    return rule_str
    

def arm_failure(founders_data_sample,feature_combination, min_sample, random_sample_size = 1):
    # This function performs association rule mining to find patterns in the data that correlate with success.
    # The main goals are to find frequent feature combinations that appear in successful founders and to generate association rules.
    # This is useful for analyzing which combinations of characteristics are most strongly associated with founder success.


    # Calculate random success probability
    real_world_prob = 89.1
    # Set the real-world probability of success (1.9%) to adjust the baseline calculation.
    # This is used to normalize the probability of success in the sample to reflect real-world expectations.
    random_failure_prob = (founders_data_sample['success'] == 0).mean() * 100

    # Calculate the random probability of success based on the percentage of successful founders in the dataset.
    # This serves as the baseline probability to compare against combinations found through association rule mining.
    adjusted_random_prob = real_world_prob
    print(f"Random probability of success (baseline): {random_failure_prob:.2f}%")


    # Filter and encode relevant columns
    categorical_data = founders_data_sample.select_dtypes(include=['int64', 'bool', 'object']).copy()
    # Select only categorical columns for encoding and analysis. This ensures that numerical features are not mistakenly included.
    

    categorical_data_encoded = pd.get_dummies(categorical_data.drop(columns=['success']), columns=categorical_data.columns.drop('success')).astype(bool)
    # Convert categorical columns into binary format using one-hot encoding.
    # The success column is excluded because it's the target variable, not an input feature.

    # Filter out positive indicators (e.g., *_0 or *_False) to focus on negative features only
    negative_columns = [col for col in categorical_data_encoded.columns if col.endswith('_0') or col.endswith('_False') or col.endswith('_nope')]
    # Filter out positive indicators (e.g., *_0, *_False, *_nope) to focus only on negative traits.
    # This ensures that only attributes indicative of negative experiences or characteristics are analyzed.
    categorical_data_encoded = categorical_data_encoded[negative_columns]


    # Use Apriori to find frequent itemsets with higher minimum support to reduce load
    frequent_itemsets = apriori(categorical_data_encoded, min_support=(min_sample / len(categorical_data_encoded)), use_colnames=True, verbose=1, max_len=feature_combination)
    # Use the Apriori algorithm to identify frequent feature combinations that meet the minimum support threshold.
    # The feature_combination parameter determines the maximum size of feature combinations (e.g., pairwise).
    if frequent_itemsets.empty:
        print("No frequent itemsets were generated. Please adjust the min_support value.")
        return

    # Filter itemsets to only include those with exactly the specified number of features
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == feature_combination)]
    # Filter frequent itemsets to include only those with exactly the specified number of features.
    # This ensures that the output matches the desired feature combination size (e.g., pairwise).

    # Calculate success probability for each itemset
    import math
    failure_probabilities = []
    likelihood_of_failure = []
    sample_counts = []
    standard_deviations = []
    confidence_intervals = []
    for _, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        # Filter the original sample to include rows that have all items in the itemset
        mask = pd.Series(True, index=founders_data_sample.index)
        # Create a boolean mask to filter the dataset for rows that match all features in the itemset.
        for item in itemset:
            if item in categorical_data_encoded.columns:
                mask &= categorical_data_encoded[item]
                filtered_data = founders_data_sample[mask]
        # Calculate success probability for the filtered data
        failure_count = (filtered_data['success'] == 0).sum()  # Count failures (success = 0)
        failure_probability = (failure_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
        # Calculate the success probability for the filtered data subset.
        # This measures the percentage of founders with the itemset features that were successful.
        likelihood = (failure_probability / random_failure_prob) if random_failure_prob > 0 else 0
        likelihood_of_failure.append(likelihood)
        sample_counts.append(len(filtered_data))
        # Calculate standard deviation based on binomial distribution
        p = failure_probability / 100
        n = len(filtered_data)
        standard_deviation = math.sqrt(p * (1 - p) / n) * 100 if n > 0 else 0
        # Calculate 95% confidence interval
        Z = 1.96  # Z-score for 95% confidence level
        margin_of_error = Z * standard_deviation
        confidence_lower = max(0, failure_probability - margin_of_error)
        confidence_upper = min(100, failure_probability + margin_of_error)
        confidence_intervals.append((confidence_lower, confidence_upper))
        failure_probabilities.append(failure_probability)
        standard_deviations.append(standard_deviation)

    frequent_itemsets['failure_probability'] = failure_probabilities
    frequent_itemsets['sample_count'] = sample_counts
    frequent_itemsets['likelihood_of_failure'] = likelihood_of_failure
    frequent_itemsets['real_world_prob'] = frequent_itemsets['failure_probability'] * (real_world_prob / random_failure_prob)
    frequent_itemsets['confidence_interval_95'] = [(round(conf[0] * (real_world_prob / random_failure_prob), 2), round(conf[1] * (real_world_prob / random_failure_prob), 2)) for conf in confidence_intervals]
    # Add calculated columns to the frequent itemsets DataFrame, including:
    # - success_probability: Percentage of success for founders with the itemset features.
    # - sample_count: Number of samples matching the itemset.
    # - likelihood_of_success: Comparison of success probability to the random baseline.
    # - real_world_prob: Normalized success probability to reflect real-world distribution.
    # - confidence_interval_95: 95% confidence interval for success probability.

    # Display the number of frequent itemsets identified
    print(f"Number of frequent itemsets identified: {len(frequent_itemsets)}")

    # Display the top 10 frequent itemsets based on success probability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(f"Top Frequent Itemsets by Failure Probability (including sample count, likelihood of failure, and 95% confidence interval):")
    if not frequent_itemsets.empty:
        top_itemsets = frequent_itemsets[['itemsets', 'failure_probability', 'sample_count', 'likelihood_of_failure', 'real_world_prob', 'confidence_interval_95']].sort_values(by='failure_probability', ascending=False).head(5)
        filtered_top_itemsets = top_itemsets[top_itemsets['sample_count'] >= 10]
        print(top_itemsets)
        random_selection = filtered_top_itemsets.sample(n=random_sample_size)
        rule_str = random_selection[['itemsets','failure_probability']]
        print(rule_str)
    
    return rule_str
    

    """
    # Generate association rules from the frequent itemsets
    print("\nGenerating association rules...")
    if frequent_itemsets.empty:
        print("No frequent itemsets available for generating association rules. Please adjust the min_support value or check the base feature.")
        return

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1, support_only=True)
    # Generate association rules from the frequent itemsets using a minimum confidence threshold of 0.1.
    # This identifies feature combinations that are predictive of success with a high likelihood.
    if rules.empty:
        print("No association rules were generated. Please adjust the parameters.")
        return

    
    # Filter rules to only include those with 'success_1' as the consequent
    success_rules = rules[rules['consequents'].apply(lambda x: 'success' in x)]

    # Display rules predicting success with lift greater than 1 (indicating better than random)
    if not success_rules.empty:
        success_rules['probability_of_success'] = success_rules['confidence'] * 100
        # Calculate the probability of success based on the confidence of the rule.
        # Confidence represents how often the rule holds true for the data subset.
        success_rules['improvement_over_random'] = success_rules['probability_of_success'] / random_success_prob
        # Calculate the improvement over random chance for each rule.
        # This helps identify rules that significantly outperform the baseline success rate.
        filtered_rules = success_rules[success_rules['improvement_over_random'] > 1]
        print("\nTop Rules Predicting Success (better than random):")
        print(filtered_rules[['antecedents', 'probability_of_success', 'support', 'lift', 'improvement_over_random']].sort_values(by='improvement_over_random', ascending=False).head(10))
    else:
        print("\nNo high-confidence rules with 'success_1' as consequent were identified that outperform random selection.")
    """



def main_negative(feature_combination, min_sample, sample_size, base_feature, feature_value, exclude_features):
    # This function performs association rule mining to find patterns in the data that correlate with success.
    # The main goals are to find frequent feature combinations that appear in successful founders and to generate association rules.
    # This is useful for analyzing which combinations of characteristics are most strongly associated with founder success.
    # Load your data
    file_path = 'founder_data_ml.csv'
    # Load the dataset containing information on founder characteristics, which includes binary features indicating different traits and a success label.
    founders_data = pd.read_csv(file_path)

    # Calculate random success probability
    real_world_prob = 1.9
    # Set the real-world probability of success (1.9%) to adjust the baseline calculation.
    # This is used to normalize the probability of success in the sample to reflect real-world expectations.
    random_failure_prob = (founders_data['success'] == 0).mean() * 100

    # Calculate the random probability of success based on the percentage of successful founders in the dataset.
    # This serves as the baseline probability to compare against combinations found through association rule mining.
    adjusted_random_prob = real_world_prob
    print(f"Random probability of failure (baseline): {random_failure_prob:.2f}%")

    # Take a smaller sample of the data to reduce computational load
    founders_data_sample = founders_data.sample(sample_size, random_state=42)
    # Take a random sample of the dataset for analysis. This helps reduce computational load while maintaining a representative subset.

 

    # Filter and encode relevant columns
    categorical_data = founders_data_sample.select_dtypes(include=['int64', 'bool', 'object']).copy()
    # Select only categorical columns for encoding and analysis. This ensures that numerical features are not mistakenly included.
    
    if exclude_features:
        features_to_exclude = exclude_features.split(',')
        categorical_data = categorical_data.drop(columns=features_to_exclude, errors='ignore')
        # If exclude_features are provided, remove them from the dataset before proceeding.
        # This allows focusing on specific features without interference from excluded characteristics.
    categorical_data_encoded_negative = pd.get_dummies(categorical_data.drop(columns=['success']), columns=categorical_data.columns.drop('success')).astype(bool)
    # Convert categorical columns into binary format using one-hot encoding.
    # The success column is excluded because it's the target variable, not an input feature.
    


    # If base_feature is provided, filter itemsets to include only those containing the base_feature and feature_value
    if base_feature and base_feature != 'None' and feature_value and feature_value != 'None':
        base_feature_value = f"{base_feature}_{feature_value}"
        encoded_columns = categorical_data_encoded_negative.columns
        matching_columns_negative = [col for col in encoded_columns if base_feature in col and (col.endswith('_0') or col.endswith('_False') or col.endswith('_nope'))]
        # If a base feature and value are provided, filter itemsets to include only those containing the base feature.
        # This allows focusing on specific characteristics of interest, such as a specific education level or work experience.
        if not matching_columns_negative:
            print(f"No columns found for the base feature '{base_feature}' with value '{feature_value}'. Please check the feature or value.")
            return
        categorical_data_encoded_negative = categorical_data_encoded_negative[matching_columns_negative]
    else:
        # Filter out negative indicators (e.g., *_0 or *_False) to focus on positive features only
        negative_columns = [col for col in categorical_data_encoded_negative.columns if col.endswith('_0') or col.endswith('_False') or col.endswith('_nope')]
        categorical_data_encoded_negative = categorical_data_encoded_negative[negative_columns]


    # Use Apriori to find frequent itemsets with higher minimum support to reduce load
    frequent_itemsets = apriori(categorical_data_encoded_negative, min_support=(min_sample / len(categorical_data_encoded_negative)), use_colnames=True, verbose=1, max_len=feature_combination)
    # Use the Apriori algorithm to identify frequent feature combinations that meet the minimum support threshold.
    # The feature_combination parameter determines the maximum size of feature combinations (e.g., pairwise).
    if frequent_itemsets.empty:
        print("No frequent itemsets were generated. Please adjust the min_support value.")
        return

    # Filter itemsets to only include those with exactly the specified number of features
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == feature_combination)]
    # Filter frequent itemsets to include only those with exactly the specified number of features.
    # This ensures that the output matches the desired feature combination size (e.g., pairwise).

    # Calculate success probability for each itemset
    import math
    failure_probabilities = []
    likelihood_of_failure = []
    sample_counts = []
    standard_deviations = []
    confidence_intervals = []
    for _, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        # Filter the original sample to include rows that have all items in the itemset
        mask = pd.Series(True, index=founders_data_sample.index)
        # Create a boolean mask to filter the dataset for rows that match all features in the itemset.
        for item in itemset:
            if item in categorical_data_encoded_negative.columns:
                mask &= categorical_data_encoded_negative[item]
                filtered_data = founders_data_sample[mask]
        # Calculate success probability for the filtered data
        failure_count = (filtered_data['success'] == 0).sum()  # Count failures (success = 0)
        failure_probability = (failure_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
        # Calculate the failure probability for the filtered data subset.
        # This measures the percentage of founders with the itemset features that failed.
        likelihood = (failure_probability / random_failure_prob) if random_failure_prob > 0 else 0
        likelihood_of_failure.append(likelihood)
        sample_counts.append(len(filtered_data))
        # Calculate standard deviation based on binomial distribution
        p = failure_probability / 100
        n = len(filtered_data)
        standard_deviation = math.sqrt(p * (1 - p) / n) * 100 if n > 0 else 0
        # Calculate 95% confidence interval
        Z = 1.96  # Z-score for 95% confidence level
        margin_of_error = Z * standard_deviation
        confidence_lower = max(0, failure_probability - margin_of_error)
        confidence_upper = min(100, failure_probability + margin_of_error)
        confidence_intervals.append((confidence_lower, confidence_upper))
        failure_probabilities.append(failure_probability)
        standard_deviations.append(standard_deviation)

    frequent_itemsets['failure_probability'] = failure_probabilities
    frequent_itemsets['sample_count'] = sample_counts
    frequent_itemsets['likelihood_of_failure'] = likelihood_of_failure
    frequent_itemsets['real_world_prob'] = frequent_itemsets['failure_probability'] * (real_world_prob / random_failure_prob)
    frequent_itemsets['confidence_interval_95'] = [(round(conf[0] * (real_world_prob / random_failure_prob), 2), round(conf[1] * (real_world_prob / random_failure_prob), 2)) for conf in confidence_intervals]
    # Add calculated columns to the frequent itemsets DataFrame, including:
    # - failure_probability: Percentage of failure for founders with the itemset features.
    # - sample_count: Number of samples matching the itemset.
    # - likelihood_of_success: Comparison of success probability to the random baseline.
    # - real_world_prob: Normalized success probability to reflect real-world distribution.
    # - confidence_interval_95: 95% confidence interval for success probability.

    # Display the number of frequent itemsets identified
    print(f"Number of frequent itemsets identified: {len(frequent_itemsets)}")

    # Display the top 10 frequent itemsets based on success probability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(f"Top {args.num_results} Frequent Itemsets by Failure Probability (including sample count, likelihood of success, and 95% confidence interval):")
    if not frequent_itemsets.empty:
        print(frequent_itemsets[['itemsets', 'failure_probability', 'sample_count', 'likelihood_of_failure', 'real_world_prob', 'confidence_interval_95']].sort_values(by='failure_probability', ascending=False).head(args.num_results))

    # Generate association rules from the frequent itemsets
    print("\nGenerating association rules...")
    if frequent_itemsets.empty:
        print("No frequent itemsets available for generating association rules. Please adjust the min_support value or check the base feature.")
        return

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1, support_only=True)
    # Generate association rules from the frequent itemsets using a minimum confidence threshold of 0.1.
    # This identifies feature combinations that are predictive of success with a high likelihood.
    if rules.empty:
        print("No association rules were generated. Please adjust the parameters.")
        return

    # Filter rules to only include those with 'success_1' as the consequent
    failure_rules = rules[rules['consequents'].apply(lambda x: 'failure' in x)]

    # Display rules predicting failure with lift greater than 1 (indicating worse than random)
    if not failure_rules.empty:
        failure_rules['probability_of_failure'] = failure_rules['confidence'] * 100
        # Calculate the probability of failure based on the confidence of the rule.
        # Confidence represents how often the rule holds true for the data subset.
        failure_rules['improvement_over_random'] = failure_rules['probability_of_failure'] / random_failure_prob
        # Calculate the improvement over random chance for each rule.
        # This helps identify rules that significantly outperform the baseline success rate.
        filtered_rules = failure_rules[failure_rules['improvement_over_random'] > 1]
        print("\nTop Rules Predicting Success (better than random):")
        print(filtered_rules[['antecedents', 'probability_of_success', 'support', 'lift', 'improvement_over_random']].sort_values(by='improvement_over_random', ascending=False).head(10))
    else:
        print("\nNo high-confidence rules with 'success_1' as consequent were identified that outperform random selection.")


def calculate_success_probability(feature_combination, founders_data_sample):
    # Calculate random success probability
    real_world_prob = 1.9
    random_success_prob = (founders_data_sample['success'] == 1).mean() * 100

    # Filter out rows with missing values for any feature in the combination
    valid_data_mask = pd.Series(True, index=founders_data_sample.index)
    for feature in feature_combination:
        feature = feature.strip()
        if feature.startswith('not_'):
            feature = feature[4:]
        if feature in founders_data_sample.columns:
            valid_data_mask &= founders_data_sample[feature].notna()
        else:
            print("Error: column not found in dataset")
    
    founders_data_sample = founders_data_sample[valid_data_mask]

    # Create mask for the specified feature combination
    mask = pd.Series(True, index=founders_data_sample.index)
    for feature in feature_combination:
        # Strip spaces from feature name
        feature = feature.strip()
        if feature in founders_data_sample.columns:
            # Handle different data types and values
            feature_mask = pd.Series(False, index=founders_data_sample.index)
            
            # Handle boolean values (including string 'True'/'False')
            if founders_data_sample[feature].dtype == bool or founders_data_sample[feature].dtype == object:
                feature_mask = (founders_data_sample[feature].astype(str).str.lower() == 'true')
            
            # Handle numeric values
            elif pd.api.types.is_numeric_dtype(founders_data_sample[feature]):
                feature_mask = (founders_data_sample[feature] > 0)
            
            # Update the overall mask with AND operation
            mask = mask & feature_mask
        elif feature.startswith('not_'):
            feature = feature[4:]
            if feature in founders_data_sample.columns:
                feature_mask = pd.Series(False, index=founders_data_sample.index)
            
            # Handle boolean values (including string 'True'/'False')
                if founders_data_sample[feature].dtype == bool or founders_data_sample[feature].dtype == object:
                    feature_mask = (founders_data_sample[feature].astype(str).str.lower() == 'false')
                
                # Handle numeric values
                elif pd.api.types.is_numeric_dtype(founders_data_sample[feature]):
                    feature_mask = (founders_data_sample[feature] == 0)
                
                # Update the overall mask with AND operation
                mask = mask & feature_mask
                
                # Handle boolean values (including string 'True'/'False')
                
    
    # Apply the filter to get samples meeting all criteria
    filtered_data = founders_data_sample[mask]
    
    # Calculate success probability
    success_count = filtered_data['success'].sum()
    success_probability = (success_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
    print(f"feature combination: {feature_combination} ,    Number of founders who succeeded: {success_count} out of {len(filtered_data)}")
    return success_probability, len(filtered_data)  # Return both probability and count

"""
def calculate_failure_probability(feature_combination, founders_data_sample):
    # Calculate random failure probability
    real_world_prob = 89.1
    random_failure_prob = (founders_data_sample['success'] == 0).mean() * 100
        # Filter out successful founders to focus on failures
    
    founders_data_sample = founders_data_sample[founders_data_sample['success'] == 0]

    # Create mask for the specified feature combination
    mask = pd.Series(True, index=founders_data_sample.index)
    for feature in feature_combination:
        # Strip spaces from feature name
        feature = feature.strip()
        if feature.startswith('not_'):
            feature = feature[4:]
            if feature in founders_data_sample.columns:
                feature_mask = pd.Series(False, index=founders_data_sample.index)
            
            # Handle boolean values (including string 'True'/'False')
                if founders_data_sample[feature].dtype == bool or founders_data_sample[feature].dtype == object:
                    # Convert to boolean if it's a string
                    if founders_data_sample[feature].dtype == object:
                        feature_mask = (founders_data_sample[feature].astype(str).str.lower() == 'true')
                    else:
                        feature_mask = (founders_data_sample[feature] == True)
                
                # Handle numeric values
                elif pd.api.types.is_numeric_dtype(founders_data_sample[feature]):
                    feature_mask = (founders_data_sample[feature] > 0)
                
                # Update the overall mask with AND operation
                mask = mask & feature_mask
                # Handle negation by getting the actual feature name after 'not_'
            else:
                continue
        else:
            continue
        
                    # Handle different data types and values

    
    # Apply the filter to get samples meeting all criteria
    filtered_data = founders_data_sample[mask]
    
    # Calculate failure probability
    failure_probability = (len(filtered_data) / len(founders_data_sample)) * 100 if len(founders_data_sample) > 0 else 0
    return failure_probability, len(filtered_data)
"""


def calculate_failure_probability(feature_combination, founders_data_sample):
    # Calculate random failure probability
    real_world_prob = 89.1
    random_failure_prob = (founders_data_sample['success'] == 0).mean() * 100
    
    # Filter out rows with missing values for any feature in the combination
    valid_data_mask = pd.Series(True, index=founders_data_sample.index)
    for feature in feature_combination:
        feature = feature.strip()
        if feature.startswith('not_'):
            feature = feature[4:]
        if feature in founders_data_sample.columns:
            valid_data_mask &= founders_data_sample[feature].notna()
    
    founders_data_sample = founders_data_sample[valid_data_mask]


    # Create mask for the specified feature combination
    mask = pd.Series(True, index=founders_data_sample.index)
    for feature in feature_combination:
        # Strip spaces from feature name
        feature = feature.strip()
        if feature.startswith('not_'):
            feature = feature[4:]
            if feature in founders_data_sample.columns:
                feature_mask = pd.Series(False, index=founders_data_sample.index)
            
            # Handle boolean values (including string 'True'/'False')
                if founders_data_sample[feature].dtype == bool or founders_data_sample[feature].dtype == object:
                    feature_mask = (founders_data_sample[feature].astype(str).str.lower() == 'false')
                
                # Handle numeric values
                elif pd.api.types.is_numeric_dtype(founders_data_sample[feature]):
                    feature_mask = (founders_data_sample[feature] == 0)
                
                # Update the overall mask with AND operation
                mask = mask & feature_mask
                # Handle negation by getting the actual feature name after 'not_'
            else:
                continue
        elif feature in founders_data_sample.columns:
            feature_mask = pd.Series(False, index=founders_data_sample.index)
            
            # Handle boolean values (including string 'True'/'False')
            if founders_data_sample[feature].dtype == bool or founders_data_sample[feature].dtype == object:
                feature_mask = (founders_data_sample[feature].astype(str).str.lower() == 'true')
            
            # Handle numeric values
            elif pd.api.types.is_numeric_dtype(founders_data_sample[feature]):
                feature_mask = (founders_data_sample[feature] > 0)
            
            # Update the overall mask with AND operation
            mask = mask & feature_mask
                    # Handle different data types and values
    
    # Apply the filter to get samples meeting all criteria
    filtered_data = founders_data_sample[mask]
        
    # Get the success values from the filtered data instead of the original data
    failed_filtered_data = filtered_data[filtered_data['success'] == 0]
    
    # Calculate failure probability
    failure_probability = (len(failed_filtered_data) / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0

    print(f"feature combination: {feature_combination} ,    Number of founders who failed: {len(failed_filtered_data)} out of {len(filtered_data)}")
    return failure_probability, len(filtered_data)


founders_data_sample = pd.read_csv('founder_data_ml.csv').sample(1000, random_state=42)
arm_success(founders_data_sample, 2, 10)
arm_failure(founders_data_sample, 2, 10)


