import numpy as np
from HeteroSymNN.Core.Nets.dense import HeteroDense
from HeteroSymNN.Backend import hardware as HW

def run_mixed_activation_demo():
    print("\n" + "="*60)
    print("🧪 MIXED ACTIVATION DEMO (Per-Neuron Heterogeneity)")
    print("="*60)
    
    nodes = [10, 5, 1]
    
    mixed_layer_config = [
        ("relu", {}),
        ("sigmoid", {}),
        ("cos(num)", {}), 
        ("num / (1 + exp(-num))", {}), 
        ("num**2", {})
    ]
    
    output_config = [("num", {})] * 1
    detailed_activations = [mixed_layer_config, output_config]
    
    print("-> Building Model with Heterogeneous Neurons...")
    model = HeteroDense(
        nodes_structure=nodes,
        detailed_activations=detailed_activations,
        batch_size=4
    )
    
    if HW.GPU_ENABLED:
        model.set_gpu_id(0)
        model._change_COMPUTATIONAL_METHOD("GPU_CUDA")
        model.change_device("GPU")
        print("✅ Compiled mixed CUDA kernel.")
    else:
        model._change_COMPUTATIONAL_METHOD("CPU_JIT")
        print("✅ Compiled mixed C++ kernel.")

    print("\n-> Testing Neuron Behavior (Input = 2.0)...")
    
    layer0 = model.layers[0]
    
    # Hack: Force weights to be diagonal so Input[i] -> Neuron[i]
    if HW.GPU_ENABLED:
        import cupy as cp
        layer0._weights = cp.zeros_like(layer0._weights)
        layer0._biases = cp.zeros_like(layer0._biases)
        for i in range(5):
            layer0._weights[i, i] = 1.0
    else:
        layer0._weights = np.zeros_like(layer0._weights)
        layer0._biases = np.zeros_like(layer0._biases)
        for i in range(5):
            layer0._weights[i, i] = 1.0

    test_input = np.full((1, 10), 2.0).astype(np.float32)
    
    print("-> Running Layer 0 Forward...")
    if HW.GPU_ENABLED:
        import cupy as cp
        gpu_in = cp.array(test_input.T)
        gpu_out = layer0.forward(gpu_in)
        activations = cp.asnumpy(gpu_out).flatten()
    else:
        out = layer0.forward(test_input.T)
        activations = out.flatten()

    print(f"\nInput Value: 2.0")
    print("-" * 30)
    print(f"Neuron 0 (ReLU):    {activations[0]:.4f}  (Expected: 2.0)")
    print(f"Neuron 1 (Sigmoid): {activations[1]:.4f}  (Expected: {1/(1+np.exp(-2)):.4f})")
    print(f"Neuron 2 (Cos):     {activations[2]:.4f}  (Expected: {np.cos(2):.4f})")
    print(f"Neuron 3 (Swish):   {activations[3]:.4f}  (Expected: {2/(1+np.exp(-2)):.4f})")
    print(f"Neuron 4 (Square):  {activations[4]:.4f}  (Expected: 4.0)")
    print("-" * 30)
    
    print("\n✅ DEMO COMPLETE: The single kernel executed 5 different math functions in parallel.")

if __name__ == "__main__":
    run_mixed_activation_demo()