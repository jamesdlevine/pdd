import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import click

def run_example():
    """
    Demonstrates the usage of context_generator_main to generate example code.

    Inputs to context_generator_main:
        - ctx (click.Context): Click context containing global options (strength, temperature, etc.)
        - prompt_file (str): Path to the .prompt file used to generate the original code.
        - code_file (str): Path to the existing source code file.
        - output (Optional[str]): Path to save the generated example. If None, uses default naming.

    Outputs of context_generator_main:
        - generated_code (str): The resulting example code string.
        - total_cost (float): The cost of the LLM operation in USD.
        - model_name (str): The name of the AI model that performed the generation.
    """
    # 1. Setup directory structure in ./output
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    prompt_path = output_dir / "math_utils_python.prompt"
    code_path = output_dir / "math_utils.py"
    example_output_path = output_dir / "math_utils_example.py"

    # 2. Create dummy input files
    prompt_path.write_text(
        "Task: Create a utility for basic math operations.\n"
        "Include a function 'add(a, b)' that returns the sum.",
        encoding="utf-8"
    )

    code_path.write_text(
        "def add(a, b):\n    return a + b",
        encoding="utf-8"
    )

    # 3. Mock the Click Context object with 'local': True to use local execution
    # In a real CLI, this is provided by the @click.group or @click.command decorators
    ctx_obj = {
        'strength': 0.7,        # LLM power (0.0 to 1.0)
        'temperature': 0.2,     # Randomness (0.0 to 1.0)
        'force': True,          # Overwrite existing files
        'quiet': False,         # Show Rich console output
        'verbose': True,        # Detailed logging
        'time': 0.5,            # Thinking time budget (0.0 to 1.0)
        'local': True           # Force local execution to avoid cloud calls
    }

    ctx = click.Context(click.Command('example'), obj=ctx_obj)
    ctx.params = {'local': True}

    # 4. Execute the main wrapper with mocked context_generator
    # Mock the heavy LLM call to avoid timeout in demo/test environments
    mock_generated_code = '''"""Example usage of math_utils module."""
from math_utils import add

# Example: Adding two numbers
result = add(3, 5)
print(f"3 + 5 = {result}")

# Example: Adding negative numbers
result = add(-10, 7)
print(f"-10 + 7 = {result}")
'''

    with patch('pdd.context_generator_main.context_generator') as mock_gen:
        mock_gen.return_value = (mock_generated_code, 0.00125, "mock-local-model")

        from pdd.context_generator_main import context_generator_main

        generated_code, cost, model = context_generator_main(
            ctx=ctx,
            prompt_file=str(prompt_path),
            code_file=str(code_path),
            output=str(example_output_path)
        )

    # 5. Display results
    print(f"--- Generation Results ---")
    print(f"Model Used: {model}")
    print(f"Total Cost: ${cost:.6f}")
    print(f"Output saved to: {example_output_path}")
    print("\nGenerated Code Snippet:")
    print("-" * 20)
    # Handle the case where generated_code might be shorter than 150 chars
    snippet = generated_code[:150] + "..." if len(generated_code) > 150 else generated_code
    print(snippet)

if __name__ == "__main__":
    # Ensure environment variables required by internal modules are present
    # (Normally these are set in the user's shell environment)
    if "NEXT_PUBLIC_FIREBASE_API_KEY" not in os.environ:
        os.environ["NEXT_PUBLIC_FIREBASE_API_KEY"] = "mock_key"

    run_example()