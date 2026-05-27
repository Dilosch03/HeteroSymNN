import numpy as np
import matplotlib.pyplot as plt

from HeteroSymNN.Core.Nets.linear_net import MLP
from HeteroSymNN.API.wrappers import Wrapper
from HeteroSymNN.API import data_transformers as utils

def run_simple_demo():
    print("--- 1. Generating Data ---")
    X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)

    print("--- 2. Creating Model (The easy way) ---")
    model = MLP(
        nodes_structure=[1, 64, 64, 1], 
        activation=("relu(x)*alfa", {"alfa": 0.1}), 
        output_activation="num", # Linear output
        batch_size=32
    )

    # The Wrapper handles normalization and data splitting for you
    trainer = Wrapper(model, work_type="reg", input_transformer=utils.MinMaxScaler())
    trainer.load_training(X, y)

    print("--- 3. Training ---")
    trainer.run_training(num_iterations=500)
    # The dictionary format is: { layer_index: [(neuron_index, constant_name, new_value), ...] }
    trainer.model.change_constants({1: [(3,"alfa",0.9), (9,"alfa",-15.0)], 0: [(3,"alfa",0.6), (18,"alfa",1.0)]})
    trainer.run_training(num_iterations=500)
    
    print("--- 4. Evaluation ---")
    metrics = trainer.regression_test_accuracy(X, y)
    print(f"R2 Score: {metrics['R2']:.4f}")
    
    # Optional: Predict
    test_val = np.array([[1.5]]) # Roughly PI/2
    pred = trainer.predict(test_val)
    print(f"Sin(1.5) Real: {np.sin(1.5):.4f}, Pred: {pred[0,0]:.4f}")
    
    # Real-time Tuning (Zero Recompile)
    # Change constant 'alfa' in layer 0, neuron 3 to 1.8, etc.
    trainer.model.change_constants({1: [(3,"alfa",-3.8), (9,"alfa",0.8)], 0: [(3,"alfa",1.8), (18,"alfa",-91.7)]})
    pred = trainer.predict(test_val)
    print(f"Sin(1.5) Real: {np.sin(1.5):.4f}, Pred: {pred[0,0]:.4f}")

    print("\n--- 5. Saving and Loading Models ---")
    trainer.save_model("my_sine_model.symnn", overwrite=True)
    print("✅ Model saved to my_sine_model.symnn")
    
    loaded_trainer = Wrapper.load_model("my_sine_model.symnn")
    print("✅ Model loaded successfully!")

    print("\n--- 6. Plotting Results ---")
    # Reset constants to a good state for plotting
    loaded_trainer.model.change_constants({1: [(3,"alfa",0.9), (9,"alfa",-15.0)], 0: [(3,"alfa",0.6), (18,"alfa",1.0)]})
    predictions = loaded_trainer.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=1, label='True Data (Sin)', alpha=0.5)
    plt.plot(X, predictions, color='red', label='Neural Net Prediction')
    plt.title("HeteroSymNN: Sine Wave Regression")
    plt.legend()
    plt.show()

run_simple_demo()