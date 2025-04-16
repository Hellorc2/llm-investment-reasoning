import pandas as pd
from predict import predict, prediction_analysis

def llm_evaluate(iterative_index):
    from llms.deepseek import get_llm_response, get_llm_response_with_history
    previous_policies = []
    for i in range(1,iterative_index):
        with open(f'iterative_training_results/iteration_{i:03d}_preprocessed.txt', 'r') as f:
            previous_policies.append(f.read())

    test_batch = pd.read_csv(f'iterative_test_data/test_batch_{iterative_index:03d}.csv')

    # Read founder profiles and combine into sample data
    founder_profiles = ""
    
    for _, row in test_batch.iterrows():
        founder_name = row['founder_name']
        linkedin = row['cleaned_founder_linkedin_data']
        crunchbase = row['cleaned_founder_cb_data']
        success = row['success']
        
        # Combine profiles into single string
        founder_data = f"Founder: {founder_name}\nLinkedIn: {linkedin}\nCrunchbase: {crunchbase}\n---\n"
        founder_profiles += founder_data
        
    # Get actual success labels
    actual_labels = test_batch['success'].tolist()

    # Convert list of policies into a formatted string with clear separation
    policies_str = ""
    for i, policy in enumerate(previous_policies):
        policies_str += f"\nIteration {i+1} Policy:\n"
        policies_str += "=" * 50 + "\n"
        policies_str += policy
        policies_str += "\n" + "=" * 50 + "\n"
    
    system_prompt = """You are an expert VC analyst. You have a set of policies for predicting startup success, and you would like to compare which one
    is the best"""
    

    user_prompt = f"""Here are the policies produced at previous iterations for predicting startup success:

    {policies_str}.

    You are also given a new list of founders and their linkedin and crunchbase profiles:
     
    {founder_profiles}.
       
    For each policy, please predict whether each founder will succeed or fail. Strictly follow the rules 
    on the policies. For each policy, return me a list of "success" or "failure" for each founder, with the whole list labelled with the policy number.
    Don't return me your reasoning, but try to remember them for my next questions.

    """

    policy_predictions = get_llm_response(system_prompt, user_prompt)
    print(policy_predictions)


    user_prompt2 = f"""Now you are given the actual success labels for the founders. Please compare the predictions of each policy with the 
    actual labels:

    {actual_labels}
    
    Which policy do you think performed the best? Consider precision as your primary metric, and recall as your secondary metric.
    Don't care about accuracy.

    Now compare all the different policies and focus on their differences. Consider the evolution of the policies over the iterations. 
    Which differences could have led to a better/worse performance? Take a careful look at the policy from the last iteration: 
    did it perform better or worse? Return me with a new set of policies that you think would incorporate the lessons learned from the previous policies.
    """

    advice = get_llm_response_with_history(system_prompt, [{"role": "user", "content": user_prompt}, 
                                                            {"role": "assistant", "content": policy_predictions}, 
                                                            {"role": "user", "content": user_prompt2}])
    print(advice)

    with open(f'logical_statements_advice.txt', 'w') as f:
        f.write(advice)
    return advice

def evaluate(iterative_index):
    import numpy as np
    from predict import get_best_models
    previous_policies = []
    previous_policies_with_metrics = []

    test_batch_file = f'test_data_iterative{iterative_index // 5}.csv'

    for i in range(iterative_index - 4,iterative_index + 1):
        with open(f'iterative_training_results/iteration_{i:03d}_preprocessed.txt', 'r') as f:
            policy = f.read()
            print(f"reading file iterative_training_results/iteration_{i:03d}_preprocessed.txt")
            previous_policies.append(policy)
            predict(test_batch_file, i, iterative = True)

            policy_with_metrics_str = f"""
                Iteration {i:03d}:
                Policy:
                {previous_policies[i-5*(iterative_index // 5)]}"""
            
            best_models_description = get_best_models([i], iterative = True)

            policy_with_metrics_str += f"""
            Best Models Description:
            {best_models_description}
            """

            previous_policies_with_metrics.append(policy_with_metrics_str)
            
    # Join all policies with metrics into one string
    policies_with_metrics_str = "\n".join(previous_policies_with_metrics)
    print(policies_with_metrics_str)
    
    from llms.deepseek import get_llm_response

    system_prompt = f"""You are an expert VC analyst. You have a set of policies for predicting startup success, and you would like to compare the different 
    policies to come up with a new policy with the best performance. Each policy consists of a list of rules in the format
     0/1, condition1, condition2, ... conditionk, probability,  where the conditions are combined using ANDs, 0/1 tells you 
     whether it's a success rule or a failure rule, and probability is the weight assigned to the rule, which has been learned from real data."""

    user_prompt = f"""Here are the policies produced at previous iterations for predicting startup success, labelled 
    with their test results when applied to a new set of founders. In each case, there are two parameters: threshold success
     and threshold failure - these determine when to predict success after one calculates the success and failure probabilities using the policy. 
     The test results are given in a list in the format threshold success, threshold failure, precision, recall:

    {policies_with_metrics_str}.

    Consider the evolution of the policies over the iterations: is it improving or getting worse?     Analyse the differences in each policy carefully
      and return me the analysis: which features may have contributed to the better/worseperformance?Treat precision as the primary metric, 
    and recall as the secondary metric. The F-scores given are the F_0.5 scores, which give more weight to precision. If you are familiar with the 
    idea of gradient descent, you can think of trying to figure out which direction to move the policy in to improve its performance.
    
    Now suggest a list of modifications to these policies that you think would improve their performance, with the above questions 
    in mind. In particular, if the policies are on a declining path, find a way to move them back on track; if one particular policy performed well,
     try to focus on its differences from the previous policies. If you have deleted some rules that you think are unhelpful, find 
    inspiration in the previous policies. DO NOT come up with new rules yourself. Keep the same number of success rules and the same number of failure rules.

    Based on your previous analysis, you are advised to:
    1. avoid judging the founder traits themselves without context - try to use less of your intuition and base your analysis more on how each rule might have affected
    the evolution of the policies.
    2. avoid focusing on the success and failure thresholds - these don't infer useful information about the policy.
    """

    advice = get_llm_response(system_prompt, user_prompt)
    print(advice)

    # Save original preprocessed statements before any modifications
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed.txt', 'r') as source:
        with open(f'iterative_training_results/iteration_{iterative_index:03d}_preprocessed_original.txt', 'w') as dest:
            dest.write(source.read())

    # Save advice to evaluation file
    with open(f'iterative_training_results/iteration_{iterative_index:03d}_advice.txt', 'w') as f:
        f.write(advice)

    return advice

    # Read founder profiles and combine into sample data
    
