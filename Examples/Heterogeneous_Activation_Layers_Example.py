import numpy as np
import matplotlib.pyplot as plt
from HeteroSymNN.Core.Nets.linear_net import HeteroLinearNet
from HeteroSymNN.Backend import hardware as HW

def run_mixed_activation_demo():
    print("\n" + "="*60)
    print("MIXED ACTIVATION DEMO (Per-Neuron Heterogeneity)")
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
    model = HeteroLinearNet(
        num_inputs=nodes[0],
        detailed_activations=detailed_activations,
        batch_size=4
    )
    
    if HW.GPU_ENABLED:
        model.set_gpu_id(0)
        model.set_backend("GPU_CUDA")
        model.to("device")
        print("Compiled mixed CUDA kernel.")
    else:
        model.set_backend("CPU_JIT")
        print("Compiled mixed C++ kernel.")

    print("\n-> Testing Neuron Behavior with range of inputs...")
    
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

    # Generate a range of inputs from -5 to 5
    x_vals = np.linspace(-5, 5, 100).astype(np.float32)
    # We need to test the 5 active neurons. So we copy x_vals into 5 columns
    # and pad the rest to 10 (since layer has 10 inputs)
    test_input = np.zeros((100, 10), dtype=np.float32)
    for i in range(5):
        test_input[:, i] = x_vals
    
    print("-> Running Layer 0 Forward...")
    if layer0.computational_method == "GPU_CUDA":
        gpu_in = model.cast_arrays(test_input.T)
        gpu_out = layer0.forward(gpu_in)
        activations = model.asnumpy(gpu_out) # Shape should be (10, 100) or similar
    else:
        out = layer0.forward(test_input.T)
        activations = out
        
    print("-> Plotting results...")
    plt.figure(figsize=(12, 8))
    
    labels = ["Neuron 0: ReLU", "Neuron 1: Sigmoid", "Neuron 2: Cosine", "Neuron 3: Swish", "Neuron 4: Square (x^2)"]
    for i in range(5):
        plt.plot(x_vals, activations[i, :], label=labels[i], linewidth=2)
        
    plt.title("Heterogeneous Layer Forward Pass (1 Layer, 5 Functions)")
    plt.xlabel("Input Value")
    plt.ylabel("Neuron Output")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDEMO COMPLETE: The single kernel executed 5 different math functions in parallel.")

if __name__ == "__main__":
    run_mixed_activation_demo()