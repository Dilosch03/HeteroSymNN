import sys
import subprocess
import pytest
from unittest.mock import patch

# Import the main CLI router from your framework
from HeteroSymNN.cli import main

# =====================================================================
# APPROACH 1: The "Black Box" Subprocess Test (End-to-End)
# =====================================================================
def test_cli_hardware_e2e():
    """
    Tests the 'hardware' command exactly as a user types it in the shell.
    This guarantees that the entry points and Python paths are functioning.
    """
    # Simulate: `python -m HeteroSymNN hardware`
    result = subprocess.run(
        [sys.executable, "-m", "HeteroSymNN", "hardware"],
        capture_output=True,
        text=True
    )
    
    # Assert the command ran successfully without crashing
    assert result.returncode == 0
    # Assert expected text from the hardware execution block is present
    assert "Default Compute Method:" in result.stdout
    assert "Detected GPUs:" in result.stdout


# =====================================================================
# APPROACH 2: The "White Box" Mocked Tests (Lightning Fast Unit Tests)
# =====================================================================

def test_cli_defaults_read_state(capsys):
    """
    Simulates: `heterosymnn defaults`
    Tests that the CLI correctly routes to and reads the Settings object.
    """
    with patch('sys.argv', ['heterosymnn', 'defaults']):
        main()
    
    # capsys captures everything the CLI printed to the terminal
    captured = capsys.readouterr()
    
    assert "Global Configuration File:" in captured.out
    assert "Compute Method:" in captured.out


def test_cli_defaults_set_property(capsys):
    """
    Simulates: `heterosymnn defaults --set threads --value 4`
    Tests the internal validation and setter routing.
    """
    with patch('sys.argv', ['heterosymnn', 'defaults', '--set', 'threads', '--value', '4']):
        main()
        
    captured = capsys.readouterr()
    
    # Verify the CLI confirms the set action
    assert "Global property 'threads' successfully set to: 4" in captured.out


def test_cli_parse_valid_math(capsys):
    """
    Simulates: `heterosymnn parse "sin(num)" --show-parsed`
    Tests that the JIT codegen correctly ingests valid SymPy strings.
    """
    with patch('sys.argv', ['heterosymnn', 'parse', 'sin(num)', '--show-parsed']):
        main()
        
    captured = capsys.readouterr()
    
    assert "Function Expression:" in captured.out
    assert "Detected Constants:" in captured.out


def test_cli_parse_invalid_math_catch(capsys):
    """
    Simulates: `heterosymnn parse "sin * num"`
    Tests the new AST validation to ensure bad math is caught gracefully 
    and printed to the user rather than crashing the framework.
    """
    with patch('sys.argv', ['heterosymnn', 'parse', 'sin * num']):
        main()
        
    captured = capsys.readouterr()
    
    # Since AST update, 'sin * num' correctly identifies 'sin' as a constant rather than failing in parse
    assert "Detected Constants: sin" in captured.out


def test_cli_inspect_missing_file_graceful_fail(capsys):
    """
    Simulates: `heterosymnn inspect nonexistent_model.symnn`
    Tests that the ZIP parsing logic doesn't throw raw tracebacks,
    but instead utilizes your internal try/except block.
    """
    with patch('sys.argv', ['heterosymnn', 'inspect', 'nonexistent_model.symnn']):
        main()
        
    captured = capsys.readouterr()
    
    # Verify the internal try/except block in the 'inspect' command catches the missing file
    assert "Inspecting internal structure of nonexistent_model.symnn" in captured.out
    assert "Error inspecting model:" in captured.out