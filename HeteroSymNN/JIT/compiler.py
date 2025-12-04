
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

from ..Backend import hardware as HW
from . import templates
from ..types import NodeConfig

class SymbolicJITCompiler:
    def __init__(self, configs: list[NodeConfig], calculation_method: Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],
                 device_id:int,mode: Literal["activation", "loss"] = "activation"):
        
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
                raise ValueError(f"ID de GPU {device_id} no es válido. GPUs disponibles: {HW.NUM_GPUS}")
            self._compile_cuda_kernels(configs)
            
            self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
            self.func_ids = self.func_ids_gpu

        elif (calculation_method == "CPU_JIT"):
            self._compile_cpp_kernels(configs)

        elif ( calculation_method == "CPU_PYTHON"):
            self._compile_py_kernels(configs)

    def _get_ccode_from_config(self, func_str, constants):
        constants = constants or {}
        num_symbol = sp.symbols('num',real=True)

        local_dict = {'e': mth.e, 'pi': mth.pi, 'tau': mth.tau, 'phi': (1 + mth.sqrt(5)) / 2}

        for var in self.main_vars:
            local_dict[str(var)] = var

        
        if constants:
            for name, val in constants.items():
                sym = sp.symbols(name)
                local_dict[name] = sym

        
        if func_str in templates.COMMON_FORMULAS:
            func_str = templates.COMMON_FORMULAS[func_str]
        else:
            # Inyección general por si se usa dentro de otra fórmula
            for name, formula_str in templates.COMMON_FORMULAS.items():
                try:
                    parsed_expr = sp.parse_expr(formula_str, local_dict=local_dict, evaluate=False)
                    local_dict[name] = parsed_expr
                except Exception: pass 

        func_expr = sp.parse_expr(func_str, local_dict=local_dict, evaluate=False)
        
        if callable(func_expr):
            func_expr = func_expr(*self.main_vars)

        subs_dict = {sp.symbols(k): v for k, v in constants.items()}
        func_expr_subbed = func_expr.subs(subs_dict)

        deriv_expr_subbed = sp.diff(func_expr_subbed, self.deriv_target).doit()

        return (func_expr_subbed, deriv_expr_subbed)
    
    def _generate_kernel_artifacts(self, configs: list[tuple[str, dict[str, float]]], 
                                 target_key: Literal["CPP","PY","GPU"],mode: Literal['string', 'lambda'],
                                 target_symbol: sp.Symbol = None, user_funcs: dict = None, 
                                 float_regex: re.Pattern = None):

        unique_funcs = {} 
        compiled_code = {} 

        for func_str, consts in configs:
            consts_key = frozenset(consts.items())
            func_key = (func_str, consts_key, target_key,self.mode)
            
            if not(func_key in unique_funcs):
                new_id = len(unique_funcs)
                unique_funcs[func_key] = new_id
                
                if ((HW.USE_KERNEL_CACHE) and (func_key in HW.KERNEL_CACHE)):
                    compiled_code[new_id] = HW.KERNEL_CACHE[func_key]
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
                        # Convertir a funciones lambda de Python
                        ccode_fwd = sp.lambdify(self.main_vars, func_expr, 'numpy')
                        ccode_bwd = sp.lambdify(self.main_vars, deriv_expr, 'numpy')
                        compiled_code[new_id] = (ccode_fwd, ccode_bwd)
                    
                    if (HW.USE_KERNEL_CACHE):
                        HW.KERNEL_CACHE[func_key] = compiled_code[new_id]
            
            self.func_ids_cpu.append(unique_funcs[func_key])
        
        if(mode == 'string'):
            fwd_cases = "\n".join([f"        case {fid}: return {code[0]};" for fid, code in compiled_code.items()])
            bwd_cases = "\n".join([f"        case {fid}: return {code[1]};" for fid, code in compiled_code.items()])
            return fwd_cases, bwd_cases
        
        return compiled_code

    def _compile_cpp_kernels(self, configs:list[tuple[str, dict[str, float]]]):

        if (HW.CPP_INSTALLED_COMPILER == None):
                raise Exception("CPP_JIT_ENABLED era True, pero CPP_COMPILER_NAME es None.")
              
        fwd_switch_cases, bwd_switch_cases =self._generate_kernel_artifacts(configs, "CPP", mode='string', user_funcs=templates.CPP_USER_FUNCS)

        # Plantilla de código C++ con OpenMP para paralelización
        if self.mode == "activation":
            # SymPy usa 'num', lo cambiamos por el nombre del argumento C++
            fwd_cases = fwd_switch_cases.replace("num", "z_val")
            bwd_cases = bwd_switch_cases.replace("num", "z_val")
            
            cpp_template = templates.CPP_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_switch_cases":fwd_cases,"bwd_switch_cases":bwd_cases})
           
        else: # LOSS
            # SymPy usa 'y_pred' y 'y_true'
            cpp_template = templates.CPP_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_switch_cases":fwd_switch_cases,"bwd_switch_cases":bwd_switch_cases})
        
        # --- Compilación JIT (la parte complicada) ---
        try:
            # Nombres de archivos temporales
            # Usamos un hash de la config para cachear la librería compilada
            import hashlib
            config_hash = hashlib.md5(json.dumps(configs,sort_keys=True).encode()+self.mode.encode()).hexdigest()
            
            temp_dir = HW.CPU_CACHE_DIR
            os.makedirs(temp_dir, exist_ok=True)
            
            lib_name = f"kernel_{self.mode}_{config_hash}"
            src_path = os.path.join(temp_dir, f"{lib_name}.cpp")
            
            extencion = "dll"
            if not(os.name in ["nt","Windows"]):
                extencion = "so"

            lib_path = os.path.join(temp_dir, f"{lib_name}."+extencion)
            if (HW.CPP_INSTALLED_COMPILER == "cl.exe"):
                compile_cmd = [
                    'cl.exe', '/O2', '/LD', # Optimizar y crear DLL
                    '/openmp', "/fp:fast",             # Habilitar OpenMP
                    '/Fe' + lib_path,       # Archivo de salida
                    '/EHsc',                # Manejo de excepciones
                    src_path
                ]
            else:
                compile_cmd = [
                    HW.CPP_INSTALLED_COMPILER, '-O3', '-shared', '-fPIC', '-fopenmp',
                    '-o',"-ffast-math", lib_path, src_path
                ]

            # Si la librería ya existe, no la re-compilamos
            if not (os.path.exists(lib_path)):
                with open(src_path, 'w') as f:
                    f.write(cpp_template)
                
                try:
                    compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        raise Exception(f"Falló la compilación C++ JIT. {compile_result.stderr}")
                except Exception:
                    compiler_path = shutil.which(HW.CPP_INSTALLED_COMPILER)
                    try:
                        dlls_dir = os.path.dirname(compiler_path)
                        os.add_dll_directory(dlls_dir)
                    except Exception:
                        os.environ['PATH'] = dlls_dir + os.pathsep + os.environ['PATH']
                    compile_result = subprocess.run(compile_cmd, check=False, capture_output=True, text=True)
                    if compile_result.returncode != 0:
                        raise Exception(f"Falló la compilación C++ JIT. {compile_result.stderr}")    

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

            # Wrappers Python -> C
            if self.mode == "activation":
                f_func = lib.forward_activation_kernel
                f_func.argtypes = [P_FLOAT, P_FLOAT, P_INT, C_INT, C_INT, C_INT]
                
                b_func = lib.backward_delta_kernel
                b_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, C_INT, C_INT, C_INT]

                def f_wrapper(z, a, n, b):
                    # z y a son arrays de numpy (float32)
                    f_func(
                        z.ctypes.data_as(P_FLOAT),
                        a.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, n, b, n * b
                    )

                def b_wrapper(z, err, d, n, b):
                    b_func(
                        z.ctypes.data_as(P_FLOAT),
                        err.ctypes.data_as(P_FLOAT),
                        d.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, n, b, n * b
                    )
            else: # LOSS
                f_func = lib.loss_kernel_fwd
                f_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, C_INT]
                
                b_func = lib.loss_kernel_bwd
                b_func.argtypes = [P_FLOAT, P_FLOAT, P_FLOAT, P_INT, C_INT]

                def f_wrapper(yp, yt, res):
                    f_func(
                        yp.ctypes.data_as(P_FLOAT),
                        yt.ctypes.data_as(P_FLOAT),
                        res.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, yp.size
                    )

                def b_wrapper(yp, yt, grad):
                    b_func(
                        yp.ctypes.data_as(P_FLOAT),
                        yt.ctypes.data_as(P_FLOAT),
                        grad.ctypes.data_as(P_FLOAT),
                        func_ids_ptr, yp.size
                    )

            self.forward_kernel = f_wrapper
            self.backward_kernel = b_wrapper

        except Exception as e:
            if (HW.WARNINGS_STRICT_MODE):
                raise (f"¡ERROR FATAL DE COMPILACIÓN C++ JIT!") from e
            else:
                full_warning = f"¡ERROR FATAL DE COMPILACIÓN C++ JIT! {e} "+"Causa probable: No se encontró un compilador C++ (g++ o cl.exe) en el PATH del sistema o falló OpenMP."
                full_warning += " Usando el kernel de Python (lento) como fallback."
                warnings.warn(full_warning)
                self._change_device("CPU_PYTHON") # Fallback al modo lento

    def _compile_py_kernels(self,configs:list[tuple[str, dict[str, float]]]):
        compiled = self._generate_kernel_artifacts(configs, "PY_LAMBDA", mode='lambda')
            
        if self.mode == "activation":
            # Kernel vectorizado optimizado para activaciones
            first_id = int(self.func_ids[0])
            is_homogeneous = all(fid == first_id for fid in self.func_ids)
            
            if is_homogeneous:
                func_fwd, func_bwd = compiled[first_id]
                def f_kernel(z, a, n, b): a[:] = func_fwd(z)
                def b_kernel(z, err, d, n, b): d[:] = err * func_bwd(z)
            else:
                def f_kernel(z, a, n, b):
                    for j in range(n): a[j,:] = compiled[self.func_ids[j]][0](z[j,:])
                def b_kernel(z, err, d, n, b):
                    for j in range(n): d[j,:] = err[j,:] * compiled[self.func_ids[j]][1](z[j,:])
        else:

            func_fwd, func_bwd = compiled[int(self.func_ids[0])]
            def f_kernel(y_p, y_t, res_vec): res_vec[:] = func_fwd(y_p, y_t)
            def b_kernel(y_p, y_t, grad_vec): grad_vec[:] = func_bwd(y_p, y_t)

        self.forward_kernel = f_kernel
        self.backward_kernel = b_kernel


    def _compile_cuda_kernels(self, configs:list[tuple[str, dict[str, float]]]):
        float_regex = re.compile(r"(\d+\.\d*([eE][+-]?\d+)?)")

        fwd_switch_cases, bwd_switch_cases = self._generate_kernel_artifacts(configs, "GPU", mode='string', 
                                                             user_funcs=templates.CUDA_USER_FUNCS, 
                                                             float_regex=float_regex)
        if self.mode == "activation":
            fwd_cases = fwd_switch_cases.replace("num", "z_val")
            bwd_cases = bwd_switch_cases.replace("num", "z_val")
            
            template = templates.CUDA_KERNEL_TEMPLATE_ACTIVATION.substitute({"fwd_cases":fwd_cases,"bwd_cases":bwd_cases})
            kernel_names = ["forward_activation_kernel", "backward_delta_kernel"]
            
        else: # LOSS
            # SymPy usa 'y_pred' y 'y_true' que coinciden con los argumentos del switch
            template = templates.CUDA_KERNEL_TEMPLATE_LOSS.substitute({"fwd_switch_cases":fwd_switch_cases,"bwd_switch_cases":bwd_switch_cases})
            kernel_names = ["loss_kernel_fwd", "loss_kernel_bwd"]

        try:
            with HW.be.cuda.Device(self.device_id):
                fwd_k = HW.be.RawKernel(template, kernel_names[0])
                bwd_k = HW.be.RawKernel(template, kernel_names[1])
        
        except Exception as e:
            error_message = f"¡ERROR FATAL DE COMPILACIÓN CUDA JIT! {e}"
            if HW.WARNINGS_STRICT_MODE:
                raise RuntimeError(error_message) from e
            else:
                full_warning = (error_message + 
                                " Causa probable: Error en la generación del kernel de CUDA o fallo de CuPy." +
                                " Usando el kernel de CPU como fallback.")
                warnings.warn(full_warning)  
                self._change_method("CPU_JIT")
                return 

        # Wrappers para invocación CUDA
        if self.mode == "activation":
            def f_k_wrapper(z, a, n, b):
                tot = n * b
                grid, block = HW._get_cuda_dims(tot, self.device_id)
                fwd_k(grid, block, (z, a, self.func_ids, n, b, tot))
            
            def b_k_wrapper(z, err, d, n, b):
                tot = n * b
                grid, block = HW._get_cuda_dims(tot, self.device_id)
                bwd_k(grid, block, (z, err, d, self.func_ids, n, b, tot))
        else:
            def f_k_wrapper(yp, yt, res):
                n = yp.size
                grid, block = HW._get_cuda_dims(n, self.device_id)
                fwd_k(grid, block, (yp, yt, res, self.func_ids, n))
            
            def b_k_wrapper(yp, yt, grad):
                n = yp.size
                grid, block = HW._get_cuda_dims(n, self.device_id)
                bwd_k(grid, block, (yp, yt, grad, self.func_ids, n))

        self.forward_kernel = f_k_wrapper
        self.backward_kernel = b_k_wrapper
            
        

    def _change_method(self,new_calculatuion_method:Literal["GPU_CUDA","CPU_JIT","CPU_PYTHON"],gpu_id:int):
        if(new_calculatuion_method != self.calculation_method):
            if ((new_calculatuion_method == "GPU_CUDA") and (HW.GPU_ENABLED)):
                if (gpu_id >= HW.NUM_GPUS):
                    raise ValueError(f"ID de GPU {gpu_id} no es válido. GPUs disponibles: {HW.NUM_GPUS}")
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
        return self.calculation_method
    
    def set_gpu_id(self,new_id:int):
        if (new_id != self.device_id):
            self.device_id = new_id
            if (self.calculation_method == "GPU_CUDA"):
                self._compile_cuda_kernels(self.activation_funcs)
                self.func_ids_gpu = HW.be.array(self.func_ids_cpu,dtype=HW.be.int32)
                self.func_ids = self.func_ids_gpu