import sys
import os
import pytest
import runpy

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Path to the Examples directory
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Examples"))

def get_example_scripts():
    """Return a list of all python files in the Examples directory."""
    if not os.path.exists(EXAMPLES_DIR):
        return []
    return [
        f for f in os.listdir(EXAMPLES_DIR)
        if f.endswith(".py") and os.path.isfile(os.path.join(EXAMPLES_DIR, f))
    ]

@pytest.mark.parametrize("script_name", get_example_scripts())
def test_example_script_runs_successfully(script_name):
    """
    Dynamically load and run each example script using runpy.
    This ensures that the scripts run from top to bottom without exceptions.
    """
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    
    # We use run_path to execute the script in its own module namespace
    # To speed up training if hardcoded, we could potentially monkeypatch num_training_iter
    # or epochs, but here we just run them as-is ensuring they pass.
    from HeteroSymNN.config import settings

    settings.debug_mode = True

    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        pytest.fail(f"Example script {script_name} failed with exception: {e}")
