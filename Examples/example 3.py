import numpy as np
import time
from HeteroSymNN.Core.Nets import ConfigurableNN
from HeteroSymNN.Backend import hardware as HW
from HeteroSymNN.Backend.tuner import estimate_batch_capacity # The tool we made

def run_hardware_demo():
    if not HW.GPU_ENABLED:
        print("❌ This demo requires a GPU.")
        return

    print("--- 1. Low Level Setup ---")
    # Huge model to stress test memory
    nodes = [128, 2048, 2048, 2048, 10] 
    
    # Manually defining identical layers for ConfigurableNN
    activations = [ [("relu", {})] * 2048 ] * 3
    activations.append( [("softmax", {})] * 10 ) # Output layer

    model = ConfigurableNN(
        nodes_structure=nodes,
        detailed_activations=activations,
        batch_size=1024 # Starting large
    )

    print("--- 2. Hardware Diagnostics ---")
    # Check VRAM before we move the model
    free_mem, _ = HW.get_gpu_memory_info(0)
    print(f"Initial VRAM Free: {free_mem / 1024**2:.2f} MB")

    print("--- 3. Moving to GPU (Float32) ---")
    model.set_gpu_id(0)
    model.change_device("GPU")
    
    # Check VRAM consumption
    free_mem_after, _ = HW.get_gpu_memory_info(0)
    print(f"VRAM after Float32 Load: {free_mem_after / 1024**2:.2f} MB")

    print("\n--- 4. QUANTIZATION (The Magic Trick) ---")
    # Compress the model to Float16
    model.quantize("float16")
    
    free_mem_quant, _ = HW.get_gpu_memory_info(0)
    print(f"VRAM after Float16 Quantization: {free_mem_quant / 1024**2:.2f} MB")
    print("✅ Model compressed successfully!")

    print("\n--- 5. Run Training Kernel ---")
    X_dummy = np.random.rand(1024, 128).astype(np.float32) # Input
    y_dummy = np.random.rand(1024, 10).astype(np.float32)  # Target

    start = time.time()
    # Explicitly calling a single train step to measure latency
    loss = model.train_step(model._CALCULATION_MANAGER.array(X_dummy.T), 
                            model._CALCULATION_MANAGER.array(y_dummy.T))
    
    # Force sync for accurate timing
    HW.be.cuda.Device(0).synchronize()
    duration = (time.time() - start) * 1000
    
    print(f"Forward+Backward Pass Time (FP16): {duration:.2f} ms")

def run_symbolic_demo():
    print("\n" + "="*40)
    print("🔮 SYMBOLIC JIT DEMO 🔮")
    print("="*40)
    
    print("Defining a completely custom activation function:")
    print("Formula: 'sin(num) * exp(-abs(num))' (The 'Damped Sine')")
    
    # 1. Define Custom Activation
    # We use a single layer network to isolate the function
    # Input: 1000 neurons -> Output: 1000 neurons
    # All 1000 neurons use this weird math function
    custom_func = "sin(num) * exp(-Abs(num))"
    
    activations = [[(custom_func, {})] * 1000] 
    
    model = ConfigurableNN(
        nodes_structure=[1000, 1000],
        detailed_activations=activations,
        batch_size=1024
    )
    
    if HW.GPU_ENABLED:
        print("-> Compiling Custom CUDA Kernel...")
        model.set_gpu_id(0)
        model.change_device("GPU")
        # This triggers the JIT compiler to write C++ code, call NVCC, and load the kernel
        model._change_COMPUTATIONAL_METHOD("GPU_CUDA") 
        print("✅ Compilation Complete! Kernel loaded to GPU.")
    else:
        print("-> Compiling Custom C++ Kernel (CPU)...")
        model._change_COMPUTATIONAL_METHOD("CPU_JIT")
        print("✅ Compilation Complete! DLL loaded.")

    # 2. Test it
    x_test = np.linspace(-5, 5, 1024).reshape(-1, 1).astype(np.float32)
    # Broadcast input to 1000 neurons
    x_input = np.tile(x_test, (1, 1000)) 
    
    print("-> Running Forward Pass...")
    # The framework will now execute the custom math on the device
    y_pred = model.forward(x_input)
    
    # Verify values (Just check the first neuron)
    y_sample = y_pred[0, 0] # Result of the first point (-5.0)
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