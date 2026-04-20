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

from ..Backend import hardware as HW
from . import codegen
from ..types import NodeConfig
from ..exceptions import CompilationWarning,FormulaParsingError,JITCompilationError,InvalidDeviceIDError,PerformanceWarning,InvalidDeviceIDError
from ..config import settings


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
            if device_id >= HW.NUM_GPUS:
                raise InvalidDeviceIDError(f"ID given ({device_id}) is greater than the number of available GPUs ({HW.NUM_GPUS})")
            self._compile_cuda_kernels(configs)
            
            self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
            self.func_ids = self.func_ids_gpu

        elif (calculation_method == "CPU_JIT"):
            self._compile_cpp_kernels(configs)

        elif ( calculation_method == "CPU_PYTHON"):
            self._compile_py_kernels(configs)

        else:
            raise ValueError("Calculation Method given isn't GPU_CUDA, CPU_JIT or CPU_PYTHON")


    def _get_ccode_from_config(self, func_str: str, constants: dict[str, float]):
        """
        Internal method that parses a string expression into SymPy expressions.
        """
        constants = constants or {}
        
        # PRO TIP: Replaced `mth.e` and `mth.pi` with `sp.E` and `sp.pi`. 
        # Using Python float math constants causes SymPy to prematurely calculate 
        # floating point values, ruining your symbolic derivatives!
        local_dict = {'e': sp.E, 'pi': sp.pi, 'tau': 2 * sp.pi, 'phi': (1 + sp.sqrt(5)) / 2}
        
        # Populate local_dict with main_vars to preserve exact `real=True` symbol instances
        for var in self.main_vars:
            local_dict[str(var)] = var

        for_subs = {}
        sorted_constants = sorted(constants)
        for id, key in enumerate(sorted_constants):
            for_subs[sp.symbols(key)] = sp.symbols(f"params[offset+{id}]")

        # 1. Exact Match Fallback (e.g., user just types "relu" with no arguments)
        if func_str in codegen.COMMON_FORMULAS:
            func_str = codegen.COMMON_FORMULAS[func_str]

        # 2. THE FIX: Create SymPy Callables for nested string parsing
        temp_callables = {}
        
        # Extract the exact symbols from local_dict (or create fallbacks)
        # This ensures we don't accidentally mix a generic `y_pred` with a `y_pred(real=True)`
        y_pred_sym = local_dict.get('y_pred', sp.symbols('y_pred', real=True))
        y_true_sym = local_dict.get('y_true', sp.symbols('y_true', real=True))
        num_sym = local_dict.get('num', sp.symbols('num', real=True))
        
        for key, expr_string in codegen.COMMON_FORMULAS.items():
            # Parse the base formula string into an expression (e.g., Max(0, num))
            base_expr = sp.parse_expr(expr_string, local_dict=local_dict)
            
            # Check variable names as strings to safely handle real=True metadata
            free_sym_names = [str(s) for s in base_expr.free_symbols]
            
            # Determine if it's a Loss function (uses y_pred/y_true) or Activation (uses num)
            if "y_pred" in free_sym_names or "y_true" in free_sym_names:
                # Create a lambda that takes two arguments and subs them in
                temp_callables[key] = lambda pred, true, e=base_expr, yp=y_pred_sym, yt=y_true_sym: e.subs({yp: pred, yt: true})
            else:
                # Create a lambda that takes one argument and subs it into 'num'
                temp_callables[key] = lambda val, e=base_expr, n=num_sym: e.subs({n: val})
        
        # Merge the lambdas into the local parsing dictionary
        local_dict = local_dict | temp_callables

        # 3. Parse the final expression
        func_expr = sp.parse_expr(func_str, local_dict=local_dict, evaluate=False)
        func_expr = func_expr.subs(for_subs)
        free_symb = func_expr.free_symbols

        # 4. Mode-specific Free Variable Validation
        if self.mode == "activation":
            x_sym, z_sym = sp.symbols("x z")
            main_vars_found = []
            
            if num_sym in free_symb:
                main_vars_found.append(num_sym)
            if (x_sym in free_symb) and not ("z" in constants.keys()):
                main_vars_found.append(x_sym)
            if (z_sym in free_symb) and not ("x" in constants.keys()):
                main_vars_found.append(z_sym)

            if len(main_vars_found) > 1:
                if settings.warning_level == "error":
                    raise ValueError(f"Function {func_str} has {', '.join([str(x) for x in main_vars_found])} as free variables. Must have only one.")
                elif settings.warning_level == "warn":
                    warnings.warn(f"Function {func_str} has {', '.join([str(x) for x in main_vars_found])} as free variables.", CompilationWarning, stacklevel=4)
            
            # Map aliases back to the main activation variable
            func_expr = func_expr.subs({x_sym: num_sym, z_sym: num_sym})
            
        elif self.mode == "loss":
            # In loss mode, multiple main variables (y_pred and y_true) are expected.
            # We bypass the "must have only one" restriction entirely.
            pass
        
        # If the parsed result is a lambda (e.g., user typed just "mse"), unpack main_vars into it
        if callable(func_expr):
            func_expr = func_expr(*self.main_vars)

        deriv_expr_subbed = sp.diff(func_expr, self.deriv_target)

        return (func_expr, deriv_expr_subbed)

    
    def _generate_kernel_artifacts(self, configs: list[NodeConfig], 
                                 target_key: Literal["CPP","PY","GPU"],mode: Literal['string', 'lambda'],
                                 user_funcs: dict = None, float_regex: re.Pattern = None):
        """
        Internal method that generates the core logic for the kernels, either as C++/CUDA code strings or Python lambdas.

        This method handles the translation from SymPy expressions to the target language and manages 
        the kernel cache to avoid re-parsing identical expressions.

        Parameters
        ----------
        configs : list[:type:`~HeteroSymNN.types.NodeConfig`]
            List of function configurations.
        target_key : Literal["CPP", "PY", "GPU"]
            Key suffix for the cache to distinguish between backends.
        mode : Literal['string', 'lambda']
            Output format: 'string' for C++/CUDA code, 'lambda' for Python functions.
        user_funcs : dict, optional
            Dictionary mapping SymPy functions to target language functions (e.g., {'sin': 'sinf'}).
        float_regex : re.Pattern, optional
            Regex to enforce float literals (e.g., 1.0 -> 1.0f) for C++/CUDA.

        Returns
        -------
        tuple[str, str] or dict
            If mode is 'string', returns (forward_cases, backward_cases) strings for a switch statement.
            If mode is 'lambda', returns a dictionary mapping IDs to (forward_func, backward_func).
        """

        unique_funcs = {} 
        compiled_code = {} 

        for func_str, consts in configs:
            consts_key = frozenset(consts.items())
            func_key = (func_str, consts_key, target_key,self.mode)
            
            if not(func_key in unique_funcs):
                new_id = len(unique_funcs)
                unique_funcs[func_key] = new_id
                
                if ((settings.use_kernel_cache) and (func_key in settings.kernel_cache)):
                    compiled_code[new_id] = settings.kernel_cache[func_key]
                else:
                    # Generar código base
                    func_expr, deriv_expr = self._get_ccode_from_config(func_str, consts)
                    
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
                    
                    if (settings.use_kernel_cache):
                        #validar si funciona
                        settings.kernel_cache[func_key] = compiled_code[new_id]

            
            self.func_ids_cpu.append(unique_funcs[func_key])
        
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
                raise RuntimeError("CPP_JIT_ENABLED is True, but CPP_COMPILER_NAME is None.")
              
        fwd_switch_cases, bwd_switch_cases =self._generate_kernel_artifacts(configs, "CPP", mode='string', user_funcs=codegen.CPP_USER_FUNCS)


        if self.mode == "activation":
            fwd_cases = fwd_switch_cases.replace("num", "z_val")
            bwd_cases = bwd_switch_cases.replace("num", "z_val")
            
            cpp_template = codegen.CPP_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_cases":fwd_cases,"bwd_cases":bwd_cases})
           
        else: # LOSS
            cpp_template = codegen.CPP_KERNEL_TEMPLATE_LOSS.substitute({"fwd_switch_cases":fwd_switch_cases,"bwd_switch_cases":bwd_switch_cases})
        

        try:
            config_hash = hashlib.md5(json.dumps(configs,sort_keys=True).encode()+self.mode.encode()).hexdigest()
            
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
                
                try:
                    compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        raise JITCompilationError(f"C++ JIT compilation failed. {compile_result.stderr}")
                except JITCompilationError:
                    compiler_path = shutil.which(HW.CPP_INSTALLED_COMPILER)
                    try:
                        dlls_dir = os.path.dirname(compiler_path)
                        os.add_dll_directory(dlls_dir)
                    except Exception:
                        os.environ['PATH'] = dlls_dir + os.pathsep + os.environ['PATH']
                    compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        raise JITCompilationError(f"C++ JIT compilation failed. {compile_result.stderr}") 

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
            if (settings.warning_level == "error"):
                raise JITCompilationError("JIT compilation fatal error!") from e
            elif (settings.warning_level == "warn"):
                full_warning = f"JIT compilation fatal error! {e} "+"Check if compiler is in the PATH variable."
                full_warning += " Using 'CPU_PYTHON' backend as fallback."
                warnings.warn(full_warning,PerformanceWarning,stacklevel=4)
                self._change_method("CPU_PYTHON") # Fallback

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
        
        except Exception as e:
            error_message = "JIT compilation fatal error!"
            if (settings.warning_level == "error"):
                raise JITCompilationError(error_message) from e
            elif (settings.warning_level == "warn"):
                full_warning = (error_message + 
                                e +
                                " Using the CPU as fallback.")
                warnings.warn(full_warning,PerformanceWarning,stacklevel=4)  
                self._change_method("CPU_JIT")
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
            
        

    def _change_method(self,new_calculatuion_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int):
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
                if (gpu_id >= HW.NUM_GPUS):
                    raise InvalidDeviceIDError(f"ID given ({gpu_id}) is greater than the number of available GPUs ({HW.NUM_GPUS})")
                self.device_id = gpu_id
                self.func_ids_cpu = []
                self.calculation_method = "GPU_CUDA"
                self._compile_cuda_kernels(self.activation_funcs)
                self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
                self.func_ids = self.func_ids_gpu
                    
            elif ((new_calculatuion_method == "CPU_JIT")and(HW.CPP_JIT_ENABLED)):
                self.func_ids_cpu = []
                self.calculation_method = "CPU_JIT"
                self._compile_cpp_kernels(self.activation_funcs)
                self.func_ids = self.func_ids_cpu
            
            elif (new_calculatuion_method == "CPU_PYTHON"):
                self.func_ids_cpu = []
                self.calculation_method = "CPU_PYTHON"
                self._compile_py_kernels(self.activation_funcs)
                self.func_ids = self.func_ids_cpu
            else:
                raise ValueError("Calculation Method is not GPU_CUDA, CPU_JIT o CPU_PYTHON")

        return self.calculation_method
    
    def set_gpu_id(self,new_id:int):
        """
        Updates the used GPU ID and recompiles CUDA kernels.

        Parameters
        ----------
        new_id : int
            The new GPU device ID.
        """
        if (new_id >= HW.NUM_GPUS):
            raise InvalidDeviceIDError(f"ID given ({new_id}) is greater than the number of available GPUs ({HW.NUM_GPUS})")
        
        if (new_id != self.device_id):
            self.device_id = new_id
            if (self.calculation_method == "GPU_CUDA"):
                self._compile_cuda_kernels(self.activation_funcs)
                self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
                self.func_ids = self.func_ids_gpu