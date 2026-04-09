import numpy as np
import time
from HeteroSymNN.Core.Nets.dense import HeteroDense
from HeteroSymNN.Backend import hardware as HW

def run_hardware_demo():
    if not HW.GPU_ENABLED:
        print("❌ This demo requires a GPU.")
        return

    print("--- 1. Low Level Setup ---")
    nodes = [128, 2048, 2048, 2048, 10] 
    
    activations = [ [("relu", {})] * 2048 ] * 3
    activations.append( [("sigmoid", {})] * 10 ) 

    model = HeteroDense(
        nodes_structure=nodes,
        detailed_activations=activations,
        batch_size=1024 
    )

    print("--- 2. Hardware Diagnostics ---")
    # Use CuPy's internal memory pool to check actual VRAM allocation
    mempool = HW.cp.get_default_memory_pool()
    print(f"Initial VRAM Used: {mempool.used_bytes() / 1024**2:.2f} MB")

    print("--- 3. Moving to GPU ---")
    model.set_gpu_id(0)
    model.change_device("GPU")
    
    print(f"VRAM after Float32 Load: {mempool.used_bytes() / 1024**2:.2f} MB")

    print("\n--- 4. Run Training Kernel ---")
    X_dummy = np.random.rand(1024, 128).astype(np.float32)
    y_dummy = np.random.rand(1024, 10).astype(np.float32)

    start = time.time()
    loss = model.train_step(model._CALCULATION_MANAGER.array(X_dummy.T), 
                            model._CALCULATION_MANAGER.array(y_dummy.T))
    
    HW.be.cuda.Device(0).synchronize()
    duration = (time.time() - start) * 1000
    
    print(f"Forward+Backward Pass Time: {duration:.2f} ms")

def run_symbolic_demo():
    print("\n" + "="*40)
    print("🔮 SYMBOLIC JIT DEMO 🔮")
    print("="*40)
    
    print("Defining a completely custom activation function:")
    print("Formula: 'sin(num) * exp(-abs(num))' (The 'Damped Sine')")
    
    custom_func = "sin(num) * exp(-Abs(num))"
    activations = [[(custom_func, {})] * 1000] 
    
    model = HeteroDense(
        nodes_structure=[1000, 1000],
        detailed_activations=activations,
        batch_size=1024
    )
    
    print("-> Setting weights to Identity and mask to 1s to test pure activation math...")
    # Hack: Force weights to be an identity matrix, biases to 0, and mask to all 1s
    # so that the input passes directly into the activation function unaltered.
    layer0 = model.layers[0]
    layer_params = {
        '_weights': np.eye(1000, dtype=np.float32),
        '_biases': np.zeros((1, 1000), dtype=np.float32),
        '_connection_mask': np.ones((1000, 1000), dtype=np.float32)
    }
    layer0.set_parameters(layer_params)

    if HW.GPU_ENABLED:
        print("-> Compiling Custom CUDA Kernel...")
        model.set_gpu_id(0)
        model.change_device("GPU")
        model._change_COMPUTATIONAL_METHOD("GPU_CUDA") 
        print("✅ Compilation Complete! Kernel loaded to GPU.")
    elif HW.CPP_JIT_ENABLED:
        print("-> Compiling Custom C++ Kernel (CPU)...")
        model._change_COMPUTATIONAL_METHOD("CPU_JIT")
        print("✅ Compilation Complete! DLL loaded.")
    else:
        print("-> Using Python Lambdas (CPU)...")
        model._change_COMPUTATIONAL_METHOD("CPU_PYTHON")

    # 2. Test it
    x_test = np.linspace(-5, 5, 1024).reshape(-1, 1).astype(np.float32)
    x_input = np.tile(x_test, (1, 1000)) 
    
    print("-> Running Forward Pass...")
    y_pred = model.predict(x_input)
    
    # Verify values
    y_sample = y_pred[0, 0] 
    x_val = -5.0
    expected = np.sin(x_val) * np.exp(-abs(x_val))
    
    print(f"   Input: {x_val:.4f}")
    print(f"   JIT Result: {y_sample:.6f}")
    print(f"   NumPy Check: {expected:.6f}")
    
    if abs(y_sample - expected) < 1e-5:
        print("✅ MATH MATCH! The JIT compiled the formula correctly.")
    else:
        print("❌ MATH MISMATCH! Something went wrong.")

if __name__ == "__main__":
    run_hardware_demo()
    run_symbolic_demo()