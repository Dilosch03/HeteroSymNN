import sympy as sp
import numpy as np
import math as mth
from typing import Literal
import re
import os
import subprocess
import ctypes
import warnings
import shutil 
import json
import hashlib
import ast

from ..Backend import hardware as HW
from ..Backend.validators import _validate_gpu_id
from . import codegen
from ..types import NodeConfig
from ..exceptions import CompilationWarning,FormulaParsingError,JITCompilationError,InvalidDeviceIDError,PerformanceWarning,BackendNotAvailableError,ComputationalMethodValueError
from ..config import settings

__all__ = ["SymbolicJITCompiler"]


class SymbolicJITCompiler:
    """
    The **SymbolicJITCompiler** is the computational heart of HeteroSymNN. It is responsible for transforming 
    high-level symbolic definitions of mathematical functions (activations and losses) into highly optimized, 
    hardware-specific executable kernels at runtime.

    This compiler bridges the gap between flexibility and performance by leveraging SymPy for symbolic 
    differentiation and code generation, and then compiling that code into:
    
    *   **CUDA Kernels (GPU_CUDA):** For massive parallelism on NVIDIA GPUs using CuPy.
    *   **C++ Shared Libraries (CPU_JIT):** For high-performance CPU execution using OpenMP and system compilers (MSVC/GCC).
    *   **Python Lambdas (CPU_PYTHON):** As a fallback for maximum compatibility.

    It handles the automatic differentiation of user-defined formulas, manages the compilation cache to avoid 
    redundant work, and provides a unified interface (`forward_kernel`, `backward_kernel`) for the rest of the 
    library to execute these functions without worrying about the underlying hardware implementation.

    Parameters
    ----------
    configs : list[:type:`~HeteroSymNN.types.NodeConfig`]
        A list of configurations defining the functions to compile. 
        For 'activation' mode, this is a list of (function_name_or_expression, constants_dict) for each node.
        For 'loss' mode, this is a list containing a single tuple with the loss expression and its constants.
    calculation_method : Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
        The target backend for compilation.
    device_id : int
        The ID of the GPU device to use if compiling for CUDA.
    mode : Literal["activation", "loss"], optional
        The type of function being compiled. Determines the kernel signature and symbolic variables used 
        ('num' for activations, 'y_pred'/'y_true' for losses). Defaults to "activation".

    Attributes
    ----------
    forward_kernel : Callable
        The compiled executable function for the forward pass.
    backward_kernel : Callable
        The compiled executable function for the backward pass (gradient calculation).
    calculation_method : str
        The current active calculation method.
    device_id : int
        The current GPU ID.

    Examples
    --------
    Although this class is primarily used internally, it can be instantiated for testing custom symbolic expressions.
    
    >>> import numpy as np
    >>> from HeteroSymNN.JIT.compiler import SymbolicJITCompiler
    >>> 
    >>> configs = [("Max(0, num)", {})]
    >>> method = "CPU_PYTHON"
    >>> mode = "activation"
    >>> 
    >>> py_compiler = SymbolicJITCompiler(
    ...     configs=configs,
    ...     calculation_method=method,
    ...     mode=mode
    ... )
    >>>
    >>> configs_2 =[("d*y_pred-y_true", {"d": 2.0})]
    >>> method_2 = "GPU_CUDA"
    >>> mode_2 = "loss"
    >>> device_id = 0
    >>> 
    >>> cuda_compiler = SymbolicJITCompiler(
    ...     configs=configs_2,
    ...     calculation_method=method_2,
    ...     device_id=device_id,
    ...     mode=mode_2
    ... )
    
    """
    def __init__(self, configs: list[NodeConfig], calculation_method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],
                 device_id:int,mode: Literal["activation", "loss"] = "activation"):
        """
        Initializes the JIT compiler and compiles the kernels for the requested backend.
        """
        
        self.calculation_method = calculation_method
        self.device_id = device_id
        self.func_ids_cpu = [] 
        self.func_ids_gpu = None 
        self.forward_kernel = None
        self.backward_kernel = None
        self.func_ids = self.func_ids_cpu
        self.activation_funcs = configs
        self.mode = mode
        self.main_vars = [sp.symbols('num', real=True)]

        if (mode == "loss"):
            self.main_vars = [sp.symbols('y_pred', real=True), sp.symbols('y_true', real=True)]

        self.deriv_target = self.main_vars[0]
        self.func_ids = self.func_ids_cpu

        if self.calculation_method == 'GPU_CUDA':
            _validate_gpu_id(device_id)
            self._compile_cuda_kernels(configs)
            

        elif (calculation_method == "CPU_JIT"):
            warnings.warn("Tried to change to use 'CPU_JIT', but currently is not available."+"Using CPU_PYTHON instead.",PerformanceWarning,stacklevel=2)
            self.calculation_method = "CPU_PYTHON"
            self._compile_py_kernels(configs)

        elif ( calculation_method == "CPU_PYTHON"):
            self._compile_py_kernels(configs)

        else:
            raise ComputationalMethodValueError("Calculation Method given isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")

    def _get_ccode_from_config(self, func_str: str, provided_constants: tuple = ()):
        """
        Internal method that parses a string expression into SymPy expressions.
        Extracts free variables as required constants and returns the base function and derivative.
        """
        local_dict = {}
        general_constants = {'e': sp.E.evalf(), 'pi': sp.pi.evalf(), 'tau': (2 * sp.pi).evalf(), 'phi': ((1 + sp.sqrt(5)) / 2).evalf()}
        local_dict.update(general_constants)
        constat_sub = []

        # Populate local_dict with main_vars to preserve exact `real=True` symbol instances
        for var in self.main_vars:
            local_dict[str(var)] = var

        if func_str in codegen.COMMON_FORMULAS:
            func_str = codegen.COMMON_FORMULAS[func_str]
        
        for con in provided_constants:
            temp = sp.symbols(con, real=True)
            local_dict[con] = temp
            constat_sub.append(temp)
            
        # Use Python's AST to safely intercept user symbols (e.g. 'beta') before SymPy 
        # accidentally misidentifies them as its own built-in functions.
        try:
            tree = ast.parse(func_str, mode='eval')
            called_names = set()
            all_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node, ast.Name):
                    all_names.add(node.id)
            
            # Anything that wasn't used as a function call is a free symbol!
            uncalled_symbols = all_names - called_names
            for sym_name in uncalled_symbols:
                if sym_name not in local_dict:
                    temp = sp.symbols(sym_name, real=True)
                    constat_sub.append(temp)
                    local_dict[sym_name] = temp
        except SyntaxError:
            pass # Invalid syntax will be caught and explained beautifully by SymPy shortly.

        temp_callables = {}
        y_pred_sym = local_dict.get('y_pred', sp.symbols('y_pred', real=True))
        y_true_sym = local_dict.get('y_true', sp.symbols('y_true', real=True))
        num_sym = local_dict.get('num', sp.symbols('num', real=True))
        
        for key, expr_string in codegen.COMMON_FORMULAS.items():
            try:
                temp_dict = local_dict.copy()
                try:
                    tree = ast.parse(expr_string, mode='eval')
                    called_names = set()
                    all_names = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            called_names.add(node.func.id)
                        elif isinstance(node, ast.Name):
                            all_names.add(node.id)
                    uncalled_symbols = all_names - called_names
                    for sym_name in uncalled_symbols:
                        if sym_name not in temp_dict:
                            temp_dict[sym_name] = sp.symbols(sym_name, real=True)
                except SyntaxError:
                    pass
                base_expr = sp.parse_expr(expr_string, local_dict=temp_dict)
            except Exception as e:
                warnings.warn(f"Error parsing expression for key '{key}': {e}")
                continue
            free_sym_names = [str(s) for s in base_expr.free_symbols]
            
            if "y_pred" in free_sym_names or "y_true" in free_sym_names:
                temp_callables[key] = lambda pred, true, e=base_expr, yp=y_pred_sym, yt=y_true_sym: e.subs({yp: pred, yt: true})
            else:
                temp_callables[key] = lambda val, e=base_expr, n=num_sym: e.subs({n: val})
        
        local_dict = local_dict | temp_callables
        
        # Safely parse the user's string and catch mathematical syntax errors (like "sin * num")
        try:
            func_expr = sp.parse_expr(func_str, local_dict=local_dict, evaluate=False)
        except (TypeError, sp.SympifyError) as e:
            if "FunctionClass" in str(e) or isinstance(e, sp.SympifyError) or "unsupported operand" in str(e):
                raise FormulaParsingError(f"Malformed function in '{func_str}'.") from e
            raise FormulaParsingError(f"Type error while parsing '{func_str}': {e}") from e
        except SyntaxError as e:
            raise FormulaParsingError(f"Invalid mathematical syntax in '{func_str}'. Please check your operators and parentheses.") from e
        except Exception as e:
            raise FormulaParsingError(f"Failed to parse the formula '{func_str}'. Original error: {e}") from e

        if callable(func_expr):
            func_expr = func_expr(*self.main_vars)

        free_symb = func_expr.free_symbols
        required_constants = set()

        if self.mode == "activation":
            x_sym, z_sym = sp.symbols("x z", real=True)
            
            # Alias Resolution: If 'num' isn't explicitly used, check if 'x' or 'z' are meant to be the main variable
            has_num = num_sym in free_symb
            if not has_num:
                if x_sym in free_symb:
                    func_expr = func_expr.subs(x_sym, num_sym)
                elif z_sym in free_symb:
                    func_expr = func_expr.subs(z_sym, num_sym)
            
            # Refresh free symbols after alias substitution
            free_symb = func_expr.free_symbols
            
            # Any remaining variable that isn't 'num' or a math constant is a required parameter
            for sym in free_symb:
                if sym != num_sym and str(sym) not in general_constants:
                    required_constants.add(str(sym))
            
        elif self.mode == "loss":
            # For loss, any symbol not y_pred, y_true, or math constants is a required parameter
            for sym in free_symb:
                if sym not in self.main_vars and str(sym) not in general_constants:
                    required_constants.add(str(sym))

        # Because we already substituted aliases for num_sym above, diffing against deriv_target works perfectly
        deriv_expr_subbed = sp.diff(func_expr, self.deriv_target)

        # Evaluate the expressions to force SymPy to resolve delayed structural math (like sqrt(-1) becoming 1.0*I)
        func_eval = func_expr.evalf()
        deriv_eval = deriv_expr_subbed.evalf()

        # Check if SymPy failed to resolve the derivative (e.g., unknown custom functions)
        if deriv_expr_subbed.has(sp.Derivative):
            raise FormulaParsingError(f"SymPy could not compute the derivative of the formula '{func_str}'. This usually happens with unknown custom functions or non-differentiable operations.")

        # Check for imaginary/complex numbers which C++ float kernels cannot handle
        if func_eval.has(sp.I) or deriv_eval.has(sp.I) or func_expr.has(sp.I) or deriv_expr_subbed.has(sp.I):
            raise FormulaParsingError(f"Formula '{func_str}' results in complex/imaginary numbers (e.g., sqrt(-1)), which are not supported by the float32 kernels.")

        # Check for infinities or NaN (e.g., division by zero literal in the formula)
        if func_eval.has(sp.zoo, sp.oo, sp.nan) or deriv_eval.has(sp.zoo, sp.oo, sp.nan) or func_expr.has(sp.zoo, sp.oo, sp.nan) or deriv_expr_subbed.has(sp.zoo, sp.oo, sp.nan):
            raise FormulaParsingError(f"Formula '{func_str}' evaluates to an invalid mathematical state (Infinity or NaN). Please check for division by zero.")

        # Apply array fetch substitutions here since we already have the keys
        for_subs = {}
        constat_sub.sort(key=lambda s: s.name)
        for id, key in enumerate(constat_sub):
            for_subs[key] = sp.symbols(f"params[offset+{id}]")
            
        func_final = func_eval.subs(for_subs)
        deriv_final = deriv_eval.subs(for_subs)

        return (func_final, deriv_final, required_constants)

    
    def _generate_kernel_artifacts(self, configs: list[NodeConfig], 
                                 target_key: Literal["CPP","PY","GPU"],mode: Literal['string', 'lambda'],
                                 user_funcs: dict = None, float_regex: re.Pattern = None):
        """
        Internal method that generates the core logic for the kernels, either as C++/CUDA code strings or Python lambdas.
        """
        unique_funcs = {} 
        compiled_code = {} 
        parsed_funcs_cache = {} # Cache parsed equations during a single compilation run
        compilation_errors = [] # Collect errors for a comprehensive report

        for idx, (func_str, consts) in enumerate(configs):
            consts = consts or {}
            consts_keys = tuple(sorted(consts.keys()))
            
            # The relative offsets depend on the full dictionary provided
            func_key = (func_str, consts_keys, target_key, self.mode)
            
            try:
                if not(func_key in unique_funcs):
                    new_id = len(unique_funcs)
                    
                    if ((settings.use_kernel_cache) and (func_key in settings.kernel_cache)):
                        compiled_code[new_id] = settings.kernel_cache[func_key]
                        unique_funcs[func_key] = new_id
                    else:
                        # 1. Parse ONLY ONCE per unique string + dictionary keys combo
                        # We pass consts_keys down so 'beta' or custom variables are shielded from SymPy built-ins
                        parse_cache_key = (func_str, consts_keys)
                        if parse_cache_key not in parsed_funcs_cache:
                            parsed_funcs_cache[parse_cache_key] = self._get_ccode_from_config(func_str, consts_keys)
                        
                        func_expr, deriv_expr, required_constants = parsed_funcs_cache[parse_cache_key]
                        
                        # 2. Validate that the current neuron provides all required constants for this formula
                        missing_consts = [c for c in required_constants if c not in consts]
                        if missing_consts:
                            raise FormulaParsingError(f"Missing constants detected in formula '{func_str}'. The variables {missing_consts} were found in the equation but not provided in the constants dictionary. If you intended for these to be mathematical functions (like sin or cos), please ensure they are called with parentheses (e.g. 'sin(num)' instead of 'sin * num').")

                        # 2.5 Reserved keyword check & Unused constant warning
                        reserved_words = {'params', 'offset', 'num', 'y_pred', 'y_true', 'z_val', 'e', 'pi', 'tau', 'phi'}
                        invalid_keys = [k for k in consts_keys if k in reserved_words]
                        if invalid_keys:
                            raise FormulaParsingError(f"Cannot use reserved keywords {invalid_keys} as constant names in formula '{func_str}'. Please rename them.")
                        
                        if settings.warning_level != "ignore":
                            unused_consts = [c for c in consts_keys if c not in required_constants]
                            if unused_consts:
                                warnings.warn(f"Node {idx} - Unused constants {unused_consts} provided for formula '{func_str}'. This needlessly consumes memory in the parameter arrays.", CompilationWarning, stacklevel=4)

                        # 3. Generate backend-specific code
                        try:
                            if (mode == 'string'):
                                func_expr = func_expr.rewrite(sp.Piecewise)
                                deriv_expr = deriv_expr.rewrite(sp.Piecewise)
                                
                                ccode_fwd = sp.printing.ccode(func_expr, user_functions=user_funcs)
                                ccode_bwd = sp.printing.ccode(deriv_expr, user_functions=user_funcs)
                                
                                if float_regex:
                                    ccode_fwd = float_regex.sub(r"\1f", ccode_fwd)
                                    ccode_bwd = float_regex.sub(r"\1f", ccode_bwd)
                                    
                                compiled_code[new_id] = (ccode_fwd, ccode_bwd)

                            elif (mode == 'lambda'):
                                p_sym = sp.symbols('params')
                                off_sym = sp.symbols('offset')

                                lambda_args = self.main_vars + [p_sym, off_sym]
                                ccode_fwd = sp.lambdify(lambda_args, func_expr, 'numpy')
                                ccode_bwd = sp.lambdify(lambda_args, deriv_expr, 'numpy')
                                compiled_code[new_id] = (ccode_fwd, ccode_bwd)
                                try:
                                    dummy_args = [np.array([0.5], dtype=np.float32) for _ in self.main_vars] + [np.array([0.5]*len(required_constants)), 0]
                                    ccode_fwd(*dummy_args)
                                    ccode_bwd(*dummy_args)
                                    
                                except NameError as e:
                                    # NameError triggers when lambdify leaves an unmapped function name in the executable string.
                                    match = re.search(r"name '(.*)' is not defined", str(e))
                                    unsupported = match.group(1) if match else "an unknown/custom function"
                                    raise FormulaParsingError(
                                        f"Formula '{func_str}' contains '{unsupported}', which is not natively supported by the 'CPU_PYTHON' (NumPy) backend."
                                    ) from e
                                    
                                except AttributeError as e:
                                    # AttributeError triggers if SymPy falls back to an object without NumPy array methods.
                                    raise FormulaParsingError(
                                        f"Formula '{func_str}' uses an operation that the 'CPU_PYTHON' (NumPy) backend cannot process. Error: {e}"
                                    ) from e
                        except Exception as e:
                            raise FormulaParsingError(f"Failed to generate executable code for '{func_str}'. SymPy found the derivative but cannot translate it to the target backend ({mode}). This usually happens with advanced math functions (like factorial or gamma) that lack direct C++/CUDA equivalents. Original error: {str(e)}") from e
                        
                        if (settings.use_kernel_cache):
                            settings.kernel_cache[func_key] = compiled_code[new_id]
                            
                        # Only map the ID if compilation was 100% successful
                        unique_funcs[func_key] = new_id

                # Only build the ID list if we haven't encountered any errors yet
                if not compilation_errors:
                    self.func_ids_cpu.append(unique_funcs[func_key])
                    
            except FormulaParsingError as e:
                # Catch the error, append to our report, and continue to the next node
                if (self.mode == "activation"):
                    compilation_errors.append(f"Node {idx}: {str(e)}")
                else:
                    compilation_errors.append(str(e))

        # Raise the final comprehensive report if any errors were accumulated
        if compilation_errors:
            error_report = f"Found {len(compilation_errors)} error(s) during kernel generation:\n" + "\n".join([f"  - {err}" for err in compilation_errors])
            raise FormulaParsingError(error_report)
        
        if(mode == 'string'):
            fwd_cases = "\n".join([f"        case {fid}: return {code[0]};" for fid, code in compiled_code.items()])
            bwd_cases = "\n".join([f"        case {fid}: return {code[1]};" for fid, code in compiled_code.items()])
            return fwd_cases, bwd_cases
        
        return compiled_code

    def _compile_cpp_kernels(self, configs:list[NodeConfig]):
        """
        Internal method to compile the symbolic expressions into a C++ shared library (.dll/.so) and load it via ctypes.

        This method generates C++ code with OpenMP pragmas for parallelism, compiles it using the 
        system's C++ compiler (MSVC or GCC), and creates Python wrappers for the exported functions.

        Parameters
        ----------
        configs : list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of function configurations to compile.
        
        Raises
        ------
        Exception
            If the C++ compiler is not found or compilation fails. If strict warnings mode is false will try with "CPU_PYTHON" backend.
        """

        if (HW.CPP_INSTALLED_COMPILER == None):
                raise JITCompilationError("CPP_JIT_ENABLED is True, but CPP_COMPILER_NAME is None.")
              
        fwd_switch_cases, bwd_switch_cases =self._generate_kernel_artifacts(configs, "CPP", mode='string', user_funcs=codegen.CPP_USER_FUNCS)


        if self.mode == "activation":
            fwd_cases = fwd_switch_cases.replace("num", "z_val")
            bwd_cases = bwd_switch_cases.replace("num", "z_val")
            
            cpp_template = codegen.CPP_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_cases":fwd_cases,"bwd_cases":bwd_cases})
           
        else: # LOSS
            cpp_template = codegen.CPP_KERNEL_TEMPLATE_LOSS.substitute({"fwd_switch_cases":fwd_switch_cases,"bwd_switch_cases":bwd_switch_cases})
        

        try:
            config_hash = hashlib.sha256(json.dumps(configs,sort_keys=True).encode()+self.mode.encode()).hexdigest()
            
            temp_dir = settings.cpu_cache_dir
            os.makedirs(temp_dir, exist_ok=True)
            
            lib_name = f"kernel_{self.mode}_{config_hash}"
            src_path = os.path.join(temp_dir, f"{lib_name}.cpp")
            
            extencion = "dll"
            if not(os.name in ["nt","Windows"]):
                extencion = "so"

            lib_path = os.path.join(temp_dir, f"{lib_name}."+extencion)
            if (HW.CPP_INSTALLED_COMPILER == "cl.exe"):
                compile_cmd = [
                    'cl.exe', '/O2', '/LD',
                    '/openmp', "/fp:fast",             
                    '/Fe' + lib_path,       
                    '/EHsc',                
                    src_path
                ]
            else:
                compile_cmd = [
                    HW.CPP_INSTALLED_COMPILER, '-O3', '-shared', '-fPIC', '-fopenmp',
                    "-ffast-math", src_path, '-o', lib_path
                ]

            if not (os.path.exists(lib_path)):
                with open(src_path, 'w') as f:
                    f.write(cpp_template)

                compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                if compile_result.returncode != 0:
                    compiler_path = shutil.which(HW.CPP_INSTALLED_COMPILER)
                    try:
                        dlls_dir = os.path.dirname(compiler_path)
                        os.add_dll_directory(dlls_dir)
                    except Exception:
                        os.environ['PATH'] = dlls_dir + os.pathsep + os.environ['PATH']
                    compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        raise JITCompilationError(f"C++ JIT compilation failed.\nCompiler Output:\n{compile_result.stderr}\nEnsure your custom formula has valid C++ syntax and MSVC/GCC is installed correctly.")

            try:
                lib = ctypes.CDLL(lib_path)
            except Exception:
                compiler_path = shutil.which(HW.CPP_INSTALLED_COMPILER)
                try:
                    dlls_dir = os.path.dirname(compiler_path)
                    os.add_dll_directory(dlls_dir)
                except Exception:
                    os.environ['PATH'] = dlls_dir + os.pathsep + os.environ['PATH']
                lib = ctypes.CDLL(lib_path)
        
            
            P_FLOAT = ctypes.POINTER(ctypes.c_float)
            P_INT = ctypes.POINTER(ctypes.c_int)
            C_INT = ctypes.c_int
            
            self.func_ids_array_np = np.array(self.func_ids_cpu, dtype=np.int32)
            func_ids_ptr = self.func_ids_array_np.ctypes.data_as(P_INT)

            if self.mode == "activation":
                f_func = lib.forward_activation_kernel
                f_func.argtypes = [P_FLOAT, P_FLOAT, P_INT, P_FLOAT, P_INT,C_INT, C_INT, C_INT]
                
                b_func = lib.backward_delta_kernel
                b_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, P_FLOAT, P_INT, C_INT, C_INT, C_INT]

                def f_wrapper(z, a, params, offset_list, n, b):
                    f_func(
                        z.ctypes.data_as(P_FLOAT),
                        a.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, params.ctypes.data_as(P_FLOAT), 
                        offset_list.ctypes.data_as(P_INT), n, b, n * b
                    )

                def b_wrapper(z, err, delta, params, offset_list, n, b):
                    b_func(
                        z.ctypes.data_as(P_FLOAT),
                        err.ctypes.data_as(P_FLOAT),
                        delta.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, params.ctypes.data_as(P_FLOAT), 
                        offset_list.ctypes.data_as(P_INT), n, b, n * b
                    )
            else: # LOSS
                f_func = lib.loss_kernel_fwd
                f_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, P_FLOAT,C_INT]
                
                b_func = lib.loss_kernel_bwd
                b_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, P_FLOAT,C_INT]

                def f_wrapper(yp, yt, res, params):
                    f_func(
                        yp.ctypes.data_as(P_FLOAT),
                        yt.ctypes.data_as(P_FLOAT),
                        res.ctypes.data_as(P_FLOAT),
                        func_ids_ptr,params.ctypes.data_as(P_FLOAT), yp.size
                    )

                def b_wrapper(yp, yt, grad, params):
                    b_func(
                        yp.ctypes.data_as(P_FLOAT),
                        yt.ctypes.data_as(P_FLOAT),
                        grad.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, params.ctypes.data_as(P_FLOAT),yp.size
                    )

            self.forward_kernel = f_wrapper
            self.backward_kernel = b_wrapper

        except Exception as e:
            full_warning = f"JIT compilation fatal error! {e}\nCheck if compiler is in the PATH variable.\nUsing 'CPU_PYTHON' backend as fallback."
            warnings.warn(full_warning,PerformanceWarning,stacklevel=4)

    def _compile_py_kernels(self,configs:list[NodeConfig]):
        """
        Compiles the symbolic expressions into Python lambda functions using `sympy.lambdify`.

        This serves as a fallback backend that works on any system with NumPy, though it is 
        significantly slower than the compiled C++ or CUDA kernels.

        Parameters
        ----------
        configs : list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of function configurations to compile.
        """
        compiled = self._generate_kernel_artifacts(configs, "PY_LAMBDA", mode='lambda')
            
        if self.mode == "activation":
            first_id = int(self.func_ids[0])
            is_homogeneous = all(fid == first_id for fid in self.func_ids)
            
            if is_homogeneous:
                func_fwd, func_bwd = compiled[first_id]
                num_params = len(self.activation_funcs[0][1])
                def f_kernel(z, a,params, offset_list, n, b): 
                    if (num_params == 0):
                        param_cols = []
                    else:
                        matrix_params = params.reshape(n, num_params)
                        param_cols = [matrix_params[:, i].reshape(-1, 1) for i in range(num_params)]
                    a[:] = func_fwd(z,param_cols,0)
                def b_kernel(z, err, d,params, offset_list,n, b):
                    if (num_params == 0):
                        param_cols = []
                    else:
                        matrix_params = params.reshape(n, num_params)
                        param_cols = [matrix_params[:, i].reshape(-1, 1) for i in range(num_params)]
                    d[:] = err * func_bwd(z,param_cols,0)
            else:
                def f_kernel(z, a, params,offset_list, n, b):
                    for j in range(n): 
                        a[j,:] = compiled[self.func_ids[j]][0](z[j,:],params,offset_list[j])
                def b_kernel(z, err, d, params,offset_list, n, b):
                    for j in range(n): 
                        d[j,:] = err[j,:] * compiled[self.func_ids[j]][1](z[j,:],params,offset_list[j])
        else:

            func_fwd, func_bwd = compiled[int(self.func_ids[0])]
            def f_kernel(y_p, y_t, res_vec,params): res_vec[:] = func_fwd(y_p, y_t,params,0)
            def b_kernel(y_p, y_t, grad_vec,params): grad_vec[:] = func_bwd(y_p, y_t,params,0)

        self.forward_kernel = f_kernel
        self.backward_kernel = b_kernel


    def _compile_cuda_kernels(self, configs:list[NodeConfig]):
        """
        Compiles the symbolic expressions into CUDA kernels using CuPy.

        This method generates CUDA C code, compiles it into a CuPy RawKernel, and sets up 
        grid/block dimensions for execution on the GPU.

        Parameters
        ----------
        configs : list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of function configurations to compile.
        
        Raises
        ------
        RuntimeError
            If CUDA compilation fails and strict mode is enabled. If strict warnings mode is false will try with "CPU_JIT" backend.
        """
        float_regex = re.compile(r"(\d+\.\d*([eE][+-]?\d+)?)")

        fwd_switch_cases, bwd_switch_cases = self._generate_kernel_artifacts(configs, "GPU", mode='string', 
                                                             user_funcs=codegen.CUDA_USER_FUNCS, 
                                                             float_regex=float_regex)

        if self.mode == "activation":
            fwd_cases = fwd_switch_cases.replace("num", "z_val")
            bwd_cases = bwd_switch_cases.replace("num", "z_val")
            
            template = codegen.CUDA_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_cases":fwd_cases,"bwd_cases":bwd_cases})
            kernel_names = ["forward_activation_kernel", "backward_delta_kernel"]
            
        else: # LOSS
            template = codegen.CUDA_KERNEL_TEMPLATE_LOSS.substitute({"fwd_switch_cases":fwd_switch_cases,"bwd_switch_cases":bwd_switch_cases})
            kernel_names = ["loss_kernel_fwd", "loss_kernel_bwd"]

        try:
            with HW.be.cuda.Device(self.device_id):
                fwd_k = HW.be.RawKernel(template, kernel_names[0])
                bwd_k = HW.be.RawKernel(template, kernel_names[1])
                fwd_k.compile()
                bwd_k.compile()
        
        except Exception as e:
            error_message = f"CUDA JIT compilation fatal error for {len(configs)} functions (Mode: {self.mode})!"
            full_warning = f"{error_message}\n{e}\nCPU is required."
            warnings.warn(full_warning,PerformanceWarning,stacklevel=4)  
            self._change_method("CPU_PYTHON")
            return 

        # Wrappers
        if self.mode == "activation":
            def f_k_wrapper(z, a, params, offset_list, n, b):
                tot = n * b
                grid, block = HW._get_cuda_dims(tot, self.device_id)
                fwd_k(grid, block, (z, a, self.func_ids, params, offset_list, n, b, tot))
            
            def b_k_wrapper(z, err, d,params,offset_list, n, b):
                tot = n * b
                grid, block = HW._get_cuda_dims(tot, self.device_id)
                bwd_k(grid, block, (z, err, d, self.func_ids,params,offset_list, n, b, tot))
        else:
            def f_k_wrapper(yp, yt, res,params):
                n = yp.size
                grid, block = HW._get_cuda_dims(n, self.device_id)
                fwd_k(grid, block, (yp, yt, res, self.func_ids,params, n))
            
            def b_k_wrapper(yp, yt, grad,params):
                n = yp.size
                grid, block = HW._get_cuda_dims(n, self.device_id)
                bwd_k(grid, block, (yp, yt, grad, self.func_ids,params, n))

        self.forward_kernel = f_k_wrapper
        self.backward_kernel = b_k_wrapper
        self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
        self.func_ids = self.func_ids_gpu
            
        

    def _change_method(self,new_calculatuion_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int = None):
        """
        Internal method to change the calculation backend.

        This triggers a recompilation of the kernels for the new backend.

        Parameters
        ----------
        new_calculatuion_method : Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            The new backend to switch to.
        gpu_id : int
            The GPU ID to use if switching to CUDA.

        Returns
        -------
        Literal["GPU_CUDA", "CPU_JIT", "CPU_PYTHON"]
            The actual calculation method set (might differ from requested if fallback occurs).
        """
        new_calculatuion_method = new_calculatuion_method.upper()
        if(new_calculatuion_method != self.calculation_method):
            if ((new_calculatuion_method == "GPU_CUDA") and (HW.GPU_ENABLED)):
                if (gpu_id is None):
                    gpu_id = self.device_id
                _validate_gpu_id(gpu_id)
                self.device_id = gpu_id
                self.func_ids_cpu = []
                self.calculation_method = "GPU_CUDA"
                self._compile_cuda_kernels(self.activation_funcs)
                self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
                self.func_ids = self.func_ids_gpu
                    
            elif ((new_calculatuion_method == "CPU_JIT")):
                warnings.warn("Tried to change to use 'CPU_JIT', but currently is not available."+"CPU_PYTHON is required.",PerformanceWarning,stacklevel=4)
                self.func_ids_cpu = []
                self.calculation_method = "CPU_PYTHON"
                self._compile_py_kernels(self.activation_funcs)
                self.func_ids = self.func_ids_cpu
            
            elif (new_calculatuion_method == "CPU_PYTHON"):
                self.func_ids_cpu = []
                self.calculation_method = "CPU_PYTHON"
                self._compile_py_kernels(self.activation_funcs)
                self.func_ids = self.func_ids_cpu
            else:
                raise ComputationalMethodValueError("Calculation Method is not GPU_CUDA, CPU_JIT o CPU_PYTHON")

        return self.calculation_method
    
    def set_gpu_id(self,new_id:int):
        """
        Updates the used GPU ID and recompiles CUDA kernels.

        Parameters
        ----------
        new_id : int
            The new GPU device ID.
        """
        _validate_gpu_id(new_id)
        
        if (new_id != self.device_id):
            self.device_id = new_id
            if (self.calculation_method == "GPU_CUDA"):
                self.func_ids_cpu = []
                self._compile_cuda_kernels(self.activation_funcs)
                self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
                self.func_ids = self.func_ids_gpu