# LLM-Powered and Reasoning-Based Investment Decision Framework

## Overview
This repository implements a novel, explainable investment decision framework that leverages LLM reasoning capabilities and automated logic-based systems to evaluate early-stage startups. The objective is to surpass the random chance of successful startup selection by 10× while ensuring the process remains transparent, editable, and grounded in natural language.

Traditional ML models, while high-performing, often operate as rigid black boxes with limited interpretability. Conversely, LLMs are flexible and expressive but struggle with precise statistical inference. This project bridges that gap by using LLM-generated reasoning logs—derived from a dataset of successful and failed founders—as the foundation for constructing structured, verifiable investment heuristics. These heuristics are then transformed into interpretable rule-based systems that can be reviewed, adjusted, and improved by human experts.

The resulting decision policies are not only explainable and auditable, but also modular—each version, whether auto-generated or expert-edited, can be backtested against historical data to evaluate precision. This framework introduces a human-in-the-loop approach to investment modelling that combines the power of LLM reasoning with the rigour of automated logic, and lays the foundation for a new class of interpretable decision tools for venture capital.

## Getting Started

### Setting Up Your Environment

1. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**
   - Copy .env.sample to create your own `.env` file
   ```bash
   cp .env.sample .env
   ```
   - Add your API keys to the `.env` file
   - Always access environment variables through the `settings` object at `core/settings.py`

## Repository Structure

```
ResearchTemplateRepo/
├── core/                  # Core configuration
├── llms/                  # LLM clients and utilities
│   ├── openai/
│   ├── anthropic/
│   ├── gemini/
├── utils/                 # Utility functions
├── .env.sample            # Template for environment variables
├── lab.ipynb              # Jupyter notebook for experiments. You can add more notebooks.
├── requirements.txt       # Project dependencies
```

## Best Practices

### Working with LLMs

1. **API Organization**
   - Use the provided modules in the llms directory for each provider
   - Add model-specific functions within the appropriate module
   - Example:
     ```python
     from llms.openai import openai_client
     
     def generate_text(prompt):
         completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Write a one-sentence bedtime story about a unicorn."
                }
            ]
        )

        return completion.choices[0].message.content
     ```

2. **Recommended Models**
   - **OpenAI**: 
     - `gpt-4o-mini` for cost-effective general use. This can do for most tasks.
     - `gpt-4o` for high-quality, general-purpose responses. Use this if you need more complex reasoning.
     - `o3-mini` for highest quality outputs (expensive). Note that you can't specify the temperature for this model.
    Please, try to refrain from other OpenAI models as they are either deprecated or not recommended or super expensive. These 3 models should be enough for almost all tasks. For embedding models, you can look up at OpenAI's documentation.

   - **Anthropic**: 
     - `claude-3-7-sonnet-latest` for most complex tasks
     - `claude-3-5-haiku-latest` for faster intelligent responses
     - `claude-3-haiku-20240307` for faster, cost-efficient responses

   - **Google Gemini**:
     - `gemini-2.0-flash` for complex reasoning
     - `gemini-2.0-flash-lite` for balanced responses
     - `gemini-1.5-flash` for fastest responses

3. **Managing API Costs**
    - Use smaller models for experimentation
    - When running llms on large data, please store the reponses at intervals in txt, csv, json or pickle files. This will help you to resume the process from the last checkpoint in case of any interruption.
    - Try to use the temperature parameter to control the randomness of the responses. This will help you to get more consistent results.

### Environment Management

1. **Adding New Environment Variables**
   - Add the variable to `Settings` class in config.py
   - Update .env.sample for documentation and also update the .env file with the new variable
   - Access through `settings` object:
        ```python
        from core import settings

        api_key = settings.my_api_key
        ```

2. **Configuration Best Practices**
   - Keep API keys and other secrets in `.env` 
   - Never commit `.env` files to version control

### Utility Functions

1. **When to Add a Utility Function**
   - For code used in multiple places
   - For complex operations that should be abstracted
   - Add to utils.py and import from utils package
   Example:
   ```python
    from utils import google_search

    results = google_search("How to train a cat")
    ```

2. **Existing Utilities**
   - `google_search`: Search Google and google search snippets
   - `number_to_money`: Format numbers as readable money strings (e.g., 1000000 -> 1M)
   - `match_strings`: Compare strings with fuzzy matching. Example (match_strings("vela", "vala", threshold=0.9) -> True)
   - `camel_split`: Split camel case strings into readable format
   - `str_to_std_datetime`: Convert string to standard datetime

## Data Management

1. **Checkpointing Long Processes**
   - Save intermediate results periodically
   - Design code to resume from checkpoints
   - Example:
     ```python
    # With the below code, we process the row in the df called "prompt" and store the response in a new column, "response".
    # In order to avoid losing the progress, we save the dataframe to a csv file every 50 rows.

    def chat_with_llm(prompt):
        ...
        return "response"
        
    df["response"] = pd.Series(dtype=str)
    for idx, row in df.iterrows():
        reponse = chat_with_llm(row["prompt"])
        df.loc[idx, "response"] = response
        if idx % 50 == 0:
            df.to_csv("df_with_responses.csv")

     ```

## Additional Tips

1. **Jupyter Notebook Best Practices**
   - Use the provided lab.ipynb for experiments
   - Enable autoreload to pick up code changes: `%reload_ext autoreload` and `%autoreload 2`
   - Document your experiments with markdown cells
   - Extract reusable code into appropriate modules
   - You can use `tqdm` for progress bars in loops

2. **Version Control**
   - Commit frequently with descriptive messages
   - Use `.gitignore` to exclude large data files.

3. **Documentation**
   - Add docstrings to all functions following the example in utilities
   - Include type hints for better IDE support and code clarity
   - Document complex algorithms and data structures

By following these practices, you'll maintain a clean, efficient research environment that's easy to understand and collaborate on with other others!
