import numpy as np
from HeteroSymNN.Core.Nets.neural_nets import ConfigurableNN
from HeteroSymNN.Backend import hardware as HW

def run_mixed_activation_demo():
    print("\n" + "="*60)
    print("🧪 MIXED ACTIVATION DEMO (Per-Neuron Heterogeneity)")
    print("="*60)
    
    # 1. Define Architecture
    # Input: 10 features
    # Hidden Layer: 5 Neurons (WE WILL MIX THESE)
    # Output: 1 Neuron
    nodes = [10, 5, 1]
    
    # 2. Define "Frankenstein" Layer
    # Neuron 0: ReLU (Standard)
    # Neuron 1: Sigmoid (Gating)
    # Neuron 2: Cosine (Periodic) - Great for catching cyclical patterns
    # Neuron 3: Custom "Swish" (x * sigmoid(x))
    # Neuron 4: Custom "Square" (x^2) - Polynomial
    
    mixed_layer_config = [
        ("relu", {}),
        ("sigmoid", {}),
        ("cos", {}), # Symbolic JIT supports 'cos' natively
        ("num / (1 + exp(-num))", {}), # Manual Swish formula
        ("num**2", {})
    ]
    
    # Output Layer: Just Linear
    output_config = [("num", {})] * 1
    
    # Combine them
    detailed_activations = [mixed_layer_config, output_config]
    
    print("-> Building Model with Heterogeneous Neurons...")
    model = ConfigurableNN(
        nodes_structure=nodes,
        detailed_activations=detailed_activations,
        batch_size=4
    )
    
    # 3. Setup Hardware
    # This works on CPU or GPU seamlessly
    if HW.GPU_ENABLED:
        model.set_gpu_id(0)
        # Trigger JIT Compilation of the mixed kernel
        model._change_COMPUTATIONAL_METHOD("GPU_CUDA")
        model.change_device("GPU")
        print("✅ Compiled mixed CUDA kernel.")
    else:
        model._change_COMPUTATIONAL_METHOD("CPU_JIT")
        print("✅ Compiled mixed C++ kernel.")

    # 4. Verify Behavior
    print("\n-> Testing Neuron Behavior (Input = 2.0)...")
    
    # Create input where all features are 2.0
    # We set weights to Identity to isolate activation function behavior
    # (This assumes we can hack the weights for the demo)
    
    # Hack: Force weights to be diagonal so Input[i] -> Neuron[i]
    # Input size 10, Layer size 5. We map first 5 inputs to 5 neurons.
    layer0 = model.layers[0]
    
    # Reset weights to zero
    if HW.GPU_ENABLED:
        import cupy as cp
        layer0.weights = cp.zeros_like(layer0.weights)
        layer0.biases = cp.zeros_like(layer0.biases)
        # Set diagonal to 1.0 (Input 0->Neuron 0, Input 1->Neuron 1...)
        for i in range(5):
            layer0.weights[i, i] = 1.0
    else:
        layer0.weights = np.zeros_like(layer0.weights)
        layer0.biases = np.zeros_like(layer0.biases)
        for i in range(5):
            layer0.weights[i, i] = 1.0

    # Input Vector: [2.0, 2.0, 2.0, 2.0, 2.0, ...]
    # Batch size = 1
    test_input = np.full((1, 10), 2.0).astype(np.float32)
    
    # Run Forward
    # We capture the output of Layer 0 directly to see the activations
    # ConfigurableNN.forward runs the whole chain, so let's just run Layer 0 manually
    # Be mindful of data transfer if on GPU
    
    print("-> Running Layer 0 Forward...")
    if HW.GPU_ENABLED:
        import cupy as cp
        gpu_in = cp.array(test_input.T)
        gpu_out = layer0.forward(gpu_in)
        activations = cp.asnumpy(gpu_out).flatten()
    else:
        out = layer0.forward(test_input.T)
        activations = out.flatten()

    # 5. Print Results
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