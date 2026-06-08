import sys
import os
import pytest

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.JIT.compiler import SymbolicJITCompiler
from HeteroSymNN.exceptions import FormulaParsingError, PerformanceWarning
from HeteroSymNN.config import settings

@pytest.fixture(autouse=True)
def _setup_teardown_settings():
    # Guarantee that we are NOT caching so every test fully parses the string
    original_cache_setting = settings.use_kernel_cache
    original_warning_level = settings.warning_level
    
    settings.use_kernel_cache = False
    settings.set_warning_level("error") # Force warnings into errors for strict testing
    
    yield
    
    # Restore user settings after tests
    settings.use_kernel_cache = original_cache_setting
    settings.set_warning_level(original_warning_level)

class TestSymbolicJITCompilerExceptions:
    """
    Test suite dedicated to ensuring the SymbolicJITCompiler correctly identifies,
    catches, and reports invalid mathematical formulas, unsupported backend translations,
    and invalid memory parameters without crashing the underlying C++/CUDA environment.
    """

    def test_malformed_syntax_error(self):
        """Test that missing parentheses or bad operators throw a clear syntax error."""
        configs = [("sin * num", {})]
        with pytest.raises(FormulaParsingError, match="Missing constants detected in formula"):
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")

    def test_complex_number_rejection(self):
        """Test that formulas resulting in imaginary numbers (sp.I) are blocked."""
        configs = [("sqrt(-1) * num", {})]
        with pytest.raises(FormulaParsingError, match="results in complex/imaginary numbers"):
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")

    def test_invalid_math_state_division_by_zero(self):
        """Test that explicit division by zero (resulting in sp.zoo/sp.oo/sp.nan) is caught."""
        configs = [("log(-num) / 0", {})]
        with pytest.raises(FormulaParsingError) as exc_info:
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")
        
        assert "evaluates to an invalid mathematical state" in str(exc_info.value)
        assert "Infinity or NaN" in str(exc_info.value)

    def test_missing_required_constants(self):
        """Test that failing to provide a dictionary value for a free variable throws an error."""
        configs = [("alpha * num + beta", {"alpha": 1.0})] # 'beta' is missing
        with pytest.raises(FormulaParsingError) as exc_info:
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")
            
        assert "Missing constants detected" in str(exc_info.value)
        assert "['beta']" in str(exc_info.value)

    def test_reserved_keyword_poisoning(self):
        """Test that users cannot overwrite JIT internal variables or math constants."""
        # 'offset' is an internal C++ variable, 'pi' is a math constant
        configs = [("offset * num + pi", {"offset": 2.0, "pi": 3.14})] 
        with pytest.raises(FormulaParsingError, match="Cannot use reserved keywords"):
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")

    def test_unsupported_backend_translation(self):
        """Test that math operations with no C++/NumPy equivalent fail gracefully."""
        configs = [("factorial(num)", {})]
        with pytest.raises(FormulaParsingError) as exc_info:
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")
            
        assert "Failed to generate executable code" in str(exc_info.value)
        assert "SymPy found the derivative but cannot translate it" in str(exc_info.value)

    def test_comprehensive_error_report(self):
        """Test that multiple broken formulas across different nodes report all at once."""
        configs = [
            ("num", {}),               # Node 0: Valid
            ("sin * num", {}),         # Node 1: Malformed syntax
            ("a * num", {}),           # Node 2: Missing constant 'a'
            ("log(-num) / 0", {})      # Node 3: Division by zero
        ]
        
        with pytest.raises(FormulaParsingError) as exc_info:
            SymbolicJITCompiler(configs, "CPU_PYTHON", 0, "activation")
            
        error_msg = str(exc_info.value)
        assert "Found 3 error(s)" in error_msg
        assert "Node 1" in error_msg
        assert "Node 2" in error_msg
        assert "Node 3" in error_msg
        assert "Node 0" not in error_msg # Node 0 shouldn't have an error

    def test_cpu_jit_disabled_error(self):
        """Test that CPU_JIT correctly throws a disabled error in strict mode."""
        configs = [("num", {})]
        with pytest.raises(PerformanceWarning, match="currently is not available"):
            SymbolicJITCompiler(configs, "CPU_JIT", 0, "activation")
