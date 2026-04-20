import numpy as np
import matplotlib.pyplot as plt

from HeteroSymNN.Core.Nets.dense import MLP
from HeteroSymNN.API.wrappers import Wrapper
from HeteroSymNN.API import data_transformers as utils

def run_simple_demo():
    print("--- 1. Generating Data ---")
    X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)

    print("--- 2. Creating Model (The easy way) ---")
    model = MLP(
        nodes_structure=[1, 64, 64, 1], 
        activation=("relu*alfa", {"alfa": 0.1}), 
        output_activation="num", # Linear output
        training_mode="mini-batch",
        batch_size=32
    )

    # The Wrapper handles normalization and data splitting for you
    trainer = Wrapper(model, work_type="reg", input_transformer=utils.MinMaxScaler())
    trainer.load_training(X, y)

    print("--- 3. Training ---")
    trainer.run_training(num_iterations=500)
    trainer.model.change_constants({1: [(3,"alfa",0.9), (9,"alfa",-15.0)], 0: [(3,"alfa",0.6), (18,"alfa",1.0)]})
    trainer.run_training(num_iterations=500)
    
    print("--- 4. Evaluation ---")
    metrics = trainer.regreccion_test_accuracy(X, y)
    print(f"R2 Score: {metrics['R2']:.4f}")
    
    # Optional: Predict
    test_val = np.array([[1.5]]) # Roughly PI/2
    pred = trainer.predict(test_val)
    print(f"Sin(1.5) Real: {np.sin(1.5):.4f}, Pred: {pred[0,0]:.4f}")
    
    # Real-time Tuning (Zero Recompile)
    trainer.model.change_constants({1: [(3,"alfa",-3.8), (9,"alfa",0.8)], 0: [(3,"alfa",1.8), (18,"alfa",-91.7)]})
    pred = trainer.predict(test_val)
    print(f"Sin(1.5) Real: {np.sin(1.5):.4f}, Pred: {pred[0,0]:.4f}")

    print("--- 5. Plotting Results ---")
    trainer.model.change_constants({1: [(3,"alfa",0.9), (9,"alfa",-15.0)], 0: [(3,"alfa",0.6), (18,"alfa",1.0)]})
    predictions = trainer.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=1, label='True Data (Sin)', alpha=0.5)
    plt.plot(X, predictions, color='red', label='Neural Net Prediction')
    plt.title("HeteroSymNN: Sine Wave Regression")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simple_demo()