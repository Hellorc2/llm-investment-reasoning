

def generate_problog_program(model: str = "openai") -> None:
    # Read statements from file
    with open('logical_statements.txt', 'r') as f:
        statements_text = f.read()
    
    system_prompt = """You are a logician trying to figure out the logic behind predicting successful/unsuccessful startups."""
    
    user_prompt = f"""
    Below are multiple sets of logical rules derived from startup reflections:

        {statements_text}
    
    Now follow the analysis and convert all argument into logical statements. Don't drop any argument. Write it in a format that can be ran by problog in python

    Then integrate all the rules into a single problog program that will be used to predict the success of a startup, when given founder traits for a new founder.
"""

    if model == "openai":
        from llms.openai import get_llm_response
    else:
        from llms.deepseek import get_llm_response
        
    result = get_llm_response(system_prompt, user_prompt)
    print(result)
    
    # Write result to policy.txt
    with open('problog_program.txt', 'w') as f:
        f.write(result)


