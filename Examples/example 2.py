import numpy as np
from HeteroSymNN.Core.Nets.neural_nets import FlexibleNN, ConfigurableNN
from HeteroSymNN.Core import initializers as InitC
from HeteroSymNN.API.wrappers import GridSearchWraper
from HeteroSymNN.Core import optimizers as OptiC
from HeteroSymNN.Backend import hardware as HW

def run_expert_demo():
    print("\n" + "="*60)
    print("🧠 EXPERT ARCHITECTURE & OPTIMIZATION DEMO")
    print("="*60)

    print("\n--- PART 1: Grid Search & Flexible Activations ---")
    print("Problem: Noisy XOR (Binary Classification)")
    
    # 1. Complex Data (XOR Problem but noisy)
    X = np.random.rand(1000, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int).reshape(-1, 1)

    # 2. Defining Hyperparameter Grid
    # We test if 'Mish' (Symbolic) is better than 'ReLU' (Standard)
    # FlexibleNN allows mixing strings ("relu") with config tuples ("mish", {beta: 1.2})
    param_grid = {
        'nodes_structure': [[2, 16, 16, 1]],
        'activation_config': [
            # Config A: Standard ReLU
            ["relu", "relu", "sigmoid"],
            # Config B: Mish with custom beta parameter (JIT Compiled)
            [("mish", {"beta": 1.2}), "relu", "sigmoid"] 
        ],
        'optimizer': [
            OptiC.AdamOptimizer(0.01),
            # SgdOptimizer is stateless, saving RAM
            OptiC.SgdOptimizer(0.1) 
        ],
        'batch_size': [64]
    }

    gs = GridSearchWraper(FlexibleNN, "class", validation_testing_split=0.2)
    gs.load_training(X.tolist(), y.tolist())

    print("-> Running Grid Search...")
    best_model, best_params, results = gs.run_training(
        static_params={'num_treaning_iter': 50},
        param_grid=param_grid,
        metric_to_optimize='Acur'
    )

    print(f"\n✅ Best Accuracy: {gs.best_score:.2%}")
    print(f"✅ Best Config: {best_params['activation_config']}")


def run_heterogeneous_demo():
    print("\n" + "="*60)
    print("🌌 PART 2: HETEROGENEOUS COMPUTE & MIXED PRECISION")
    print("="*60)
    
    if not HW.GPU_ENABLED:
        print("⚠️ GPU needed for full heterogeneous demo. Skipping.")
        return

    print("Scenario: A large pipeline where one layer needs CPU logic")
    print("and the heavy lifting happens on GPU in Float16.")

    # 1. Define a Hybrid Model manually
    # Layer 0 (Input Processing): CPU Float32
    # Layer 1 (Heavy Compute): GPU Float16 (Tensor Cores)
    # Layer 2 (Output): GPU Float32 (Precision)
    
    nodes = [128, 1024, 1024, 10]
    activations = [
        [("tanh", {})] * 1024, # L0
        [("relu", {})] * 1024, # L1
        [("softmax", {})] * 10 # L2
    ]
    
    model = ConfigurableNN(nodes, activations, batch_size=256)
    
    # 2. Configure Heterogeneity
    print("\n-> Configuring Hardware Distribution:")
    
    # Layer 0: Force CPU
    L0 = model.layers[0]
    L0._change_COMPUTATIONAL_METHOD("CPU_PYTHON")
    print(f"   Layer 0: {L0.COMPUTATIONAL_METHOD} (Host Memory)")

    # Layer 1: GPU + Float16 (Quantization)
    L1 = model.layers[1]
    L1.set_gpu_id(0)
    L1._change_COMPUTATIONAL_METHOD("GPU_CUDA")
    
    # Manual quantization for just this layer
    # (In a real app, you'd use model.quantize, but this shows control)
    target_type = HW.get_dtype("float16", "GPU")
    L1.weights = L1.weights.astype(target_type)
    L1._DEFAULT_FLOAT_TYPE = target_type
    # Force JIT Recompile for __half
    L1._act_funcions_manager.dtype = target_type
    L1._act_funcions_manager._compile_for_current_method()
    print(f"   Layer 1: {L1.COMPUTATIONAL_METHOD} (FP16 Tensor Cores)")

    # Layer 2: GPU + Float32
    L2 = model.layers[2]
    L2.set_gpu_id(0)
    L2._change_COMPUTATIONAL_METHOD("GPU_CUDA")
    print(f"   Layer 2: {L2.COMPUTATIONAL_METHOD} (FP32 Precision)")

    # 3. Execute
    print("\n-> Running Hybrid Forward Pass...")
    # Input starts on CPU
    dummy_input = np.random.rand(256, 128).astype(np.float32)
    
    try:
        start = time.time()
        # The framework handles the PCIe transfers automatically:
        # Host(L0) -> PCIe -> Device(L1) -> Device(L2) -> Host(Result)
        output = model.forward(dummy_input)
        
        # Sync to measure real time
        HW.be.cuda.Device(0).synchronize()
        duration = (time.time() - start) * 1000
        
        print(f"✅ Execution Successful!")
        print(f"   Output Shape: {output.shape}")
        print(f"   Total Latency: {duration:.2f} ms (Includes PCIe transfers)")
        
    except Exception as e:
        print(f"❌ Execution Failed: {e}")

if __name__ == "__main__":
    # Needed imports for timing inside the function if copied separately
    import time 
    
    run_expert_demo()
    run_heterogeneous_demo()