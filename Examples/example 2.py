import numpy as np
import time
from HeteroSymNN.Core.Nets.dense import Dense, HeteroDense
from HeteroSymNN.API.wrappers import Wrapper, GridSearchManager
from HeteroSymNN.Backend import hardware as HW


def run_expert_demo():
    print("\n" + "="*60)
    print("🧠 EXPERT ARCHITECTURE & OPTIMIZATION DEMO")
    print("="*60)

    print("\n--- PART 1: Grid Search via Composition ---")
    print("Problem: Noisy XOR (Binary Classification)")
    
    # 1. Complex Data (XOR Problem but noisy)
    X = np.random.rand(1000, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int).reshape(-1, 1)

    # 2. Build the Template Wrapper (The Blueprint)
    template_model = Dense(
        nodes_structure=[2, 16, 16, 1],
        activation_config=["relu", "relu", "sigmoid"],
        training_mode="mini-batch",
        batch_size=64
    )
    template_wrapper = Wrapper(template_model, work_type="class")

    # 3. Defining Hyperparameter Grid
    # We mutate the exact string keys found in Wrapper.get_config()
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'batch_size': [32, 64]
    }

    # 4. Instantiate the Orchestrator
    gs = GridSearchManager(template_wrapper, param_grid, validation_split=0.2)
    gs.load_data(X.tolist(), y.tolist())

    print("-> Running Grid Search Clones in RAM...")
    best_wrapper, best_params, results = gs.execute_search(metric_to_optimize='Acur')

    print(f"\n✅ Best Accuracy: {gs.best_score:.2%}")
    print(f"✅ Best Config: {best_params}")


def run_heterogeneous_demo():
    print("\n" + "="*60)
    print("🌌 PART 2: HETEROGENEOUS COMPUTE")
    print("="*60)
    
    if not HW.GPU_ENABLED:
        print("⚠️ GPU needed for full heterogeneous demo. Skipping.")
        return

    print("Scenario: A large pipeline where one layer needs CPU logic")
    print("and the heavy lifting happens on GPU.")

    # 1. Define a Hybrid Model manually
    nodes = [128, 1024, 1024, 10]
    activations = [
        [("tanh(num)", {})] * 1024, # L0
        [("relu", {})] * 1024, # L1
        [("sigmoid", {})] * 10 # L2
    ]
    
    model = HeteroDense(nodes, activations, batch_size=256)
    
    # 2. Configure Heterogeneity
    print("\n-> Configuring Hardware Distribution:")
    
    # Layer 0: Force CPU
    L0 = model.layers[0]
    L0._change_COMPUTATIONAL_METHOD("CPU_PYTHON")
    print(f"   Layer 0: {L0.computational_method} (Host Memory)")

    # Layer 1: GPU
    L1 = model.layers[1]
    L1.set_gpu_id(0)
    L1._change_COMPUTATIONAL_METHOD("GPU_CUDA")
    print(f"   Layer 1: {L1.computational_method} (Device Memory)")

    # Layer 2: GPU
    L2 = model.layers[2]
    L2.set_gpu_id(0)
    L2._change_COMPUTATIONAL_METHOD("GPU_CUDA")
    print(f"   Layer 2: {L2.computational_method} (Device Memory)")

    # 3. Execute
    print("\n-> Running Hybrid Forward Pass...")
    dummy_input = np.random.rand(256, 128).astype(np.float32)
    
    try:
        start = time.time()
        # The framework handles the PCIe transfers automatically
        output = model.forward(dummy_input)
        
        HW.be.cuda.Device(0).synchronize()
        duration = (time.time() - start) * 1000
        
        print(f"✅ Execution Successful!")
        print(f"   Output Shape: {output.shape}")
        print(f"   Total Latency: {duration:.2f} ms (Includes PCIe transfers)")
        
    except Exception as e:
        print(f"❌ Execution Failed: {e}")

if __name__ == "__main__":
    run_expert_demo()
    run_heterogeneous_demo()