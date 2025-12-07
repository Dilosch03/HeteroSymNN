from string import Template

CUDA_USER_FUNCS = {
    # Funciones básicas
    "exp": "expf",
    "log": "logf",
    "Pow": "powf",  
    "sqrt": "sqrtf", 
    "Abs": "fabsf",  
    
    # Trigonométricas
    "sin": "sinf",
    "cos": "cosf",
    "tan": "tanf",
    
    # Trigonométricas inversas
    "asin": "asinf",
    "acos": "acosf",
    "atan": "atanf",
    "atan2": "atan2f",
    
    # Hiperbólicas
    "sinh": "sinhf",
    "cosh": "coshf",
    "tanh": "tanhf",

    # --- FUNCIONES AVANZADAS ---
    "Max": "fmaxf",          # ReLU usa esto
    "Heaviside": "heavisidef", # Derivada de ReLU
    "erf": "erff"            # Necesaria para GELU
}

CPP_USER_FUNCS = {
    "exp": "exp",
    "log": "log",
    "Pow": "pow",  
    "sqrt": "sqrt", 
    "Abs": "fabs",  
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "Max": "fmax", 
    "Heaviside": "heaviside",
    "erf": "erf"
}

COMMON_FORMULAS = {
    'relu': "Max(0.0, num)",
    'sigmoid': "1 / (1 + exp(-num))",
    'swish': "num / (1 + exp(-num))",
    "SiLU": "num / (1 + exp(-num))",
    'softplus': "log(1 + exp(num))",
    'mish': "num * tanh(log(1 + exp(num)))",
    'gelu': "0.5 * num * (1 + erf(num / sqrt(2.0)))",

    'mse': "(y_pred - y_true)**2",
    'mae': "Abs(y_pred - y_true)",
    'huber': "Piecewise((0.5 * (y_pred - y_true)**2, Abs(y_pred - y_true) <= 1.0), (Abs(y_pred - y_true) - 0.5, True))",
    'bce': "-(y_true * log(y_pred + 1e-7) + (1 - y_true) * log(1 - y_pred + 1e-7))"
}

CUDA_KERNEL_TEMPLATE_ACTIVATION = Template("""
            __device__ float apply_fwd(int id, float z_val,float* params,int offset) { switch(id){ $fwd_cases default: return z_val; } }
            __device__ float apply_bwd(int id, float z_val,float* params,int offset) { switch(id){ $bwd_cases default: return 1.0f; } }
            
            extern "C" __global__ void forward_activation_kernel(float* z, float* a, int* ids, float* params, int* offset_list, int n, int b, int tot) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < tot) {
                    int node_idx = i / b;
                    int offset = offset_list[node_idx];
                    a[i] = apply_fwd(ids[node_idx], z[i],params,offset);
                }
            }
            extern "C" __global__ void backward_delta_kernel(float* z, float* err, float* d, int* ids, float* params, int* offset_list, int n, int b, int tot) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < tot) {
                    int node_idx = i / b;
                    int offset = offset_list[node_idx];
                    d[i] = err[i] * apply_bwd(ids[node_idx], z[i],params,offset);
                }
            }
            """)

CUDA_KERNEL_TEMPLATE_LOSS = Template("""
            __device__ float loss_fwd(int id, float y_pred, float y_true, float* params, int offset) { switch(id){ $fwd_switch_cases default: return 0.0f; } }
            __device__ float loss_bwd(int id, float y_pred, float y_true, float* params, int offset) { switch(id){ $bwd_switch_cases default: return 0.0f; } }
            
            extern "C" __global__ void loss_kernel_fwd(float* yp, float* yt, float* res, int* ids, float* params, int n) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n) res[i] = loss_fwd(ids[0], yp[i], yt[i], params, 0);
            }
            extern "C" __global__ void loss_kernel_bwd(float* yp, float* yt, float* grad, int* ids, float* params, int n) {
                int i = threadIdx.x + blockIdx.x * blockDim.x;
                if (i < n) grad[i] = loss_bwd(ids[0], yp[i], yt[i], params, 0);
            }
            """)

CPP_KERNEL_TEMPLATE_ACTIVATION = Template("""
            #include <cmath>
            #include <omp.h>

            // Helpers
            inline float apply_fwd(int id, float z_val, float* params, int offset) { switch(id){ $fwd_cases default: return z_val; } }
            inline float apply_bwd(int id, float z_val, float* params, int offset) { switch(id){ $bwd_cases default: return 1.0f; } }

            extern "C" {
                void forward_activation_kernel(float* z_vec, float* a_vec, int* func_ids, float* params, int* offset_list, int num_nodes, int batch_size, int total_elements) {
                    #pragma omp parallel for
                    for (int i = 0; i < total_elements; ++i) {
                        int node_index = i / batch_size;
                        int offset = offset_list[node_index];
                        a_vec[i] = apply_fwd(func_ids[node_index], z_vec[i], params, offset);
                    }
                }
                
                void backward_delta_kernel(float* z_vec, float* error_sum, float* delta_vec, int* func_ids, float* params, int* offset_list, int num_nodes, int batch_size, int total_elements) {
                    #pragma omp parallel for
                    for (int i = 0; i < total_elements; ++i) {
                        int node_index = i / batch_size;
                        int offset = offset_list[node_index];
                        delta_vec[i] = error_sum[i] * apply_bwd(func_ids[node_index], z_vec[i], params, offset);
                    }
                }
            }
            """)

CPP_KERNEL_TEMPLATE_LOSS = Template("""
            #include <cmath>
            #include <omp.h>

            // Helpers
            inline float loss_fwd(int id, float y_pred, float y_true, float* params, int offset) { switch(id){ $fwd_switch_cases default: return 0.0f; } }
            inline float loss_bwd(int id, float y_pred, float y_true, float* params, int offset) { switch(id){ $bwd_switch_cases default: return 0.0f; } }

            extern "C" {
                void loss_kernel_fwd(float* yp, float* yt, float* res, int* ids, float* params, int n) {
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        res[i] = loss_fwd(ids[0], yp[i], yt[i],params,0);
                    }
                }
                
                void loss_kernel_bwd(float* yp, float* yt, float* grad, int* ids, float* params, int n) {
                    #pragma omp parallel for
                    for (int i = 0; i < n; ++i) {
                        grad[i] = loss_bwd(ids[0], yp[i], yt[i],params,0);
                    }
                }
            }
            """)