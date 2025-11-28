import numpy as np
import matplotlib.pyplot as plt


from HeteroSymNN.Core.Nets.neural_nets import SimpleNN
from HeteroSymNN.API.wrappers import Wraper

def run_simple_demo():
    print("--- 1. Generating Data (Sine Wave) ---")
    X = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
    y = np.sin(X)

    print("--- 2. Creating Model (The easy way) ---")

    model = SimpleNN(
        nodes_structure=[1, 64, 64, 1], 
        activation="relu", 
        output_activation="num", # Linear output
        training_mode="mini-batch",
        batch_size=32
    )

    # The Wrapper handles normalization and data splitting for you
    trainer = Wraper(model, work_type="reg", normalize_inputs=True)
    trainer.load_training(X, y)

    print("--- 3. Training ---")
    # 'mini-batch' is the default we set earlier, so this is fast automatically
    trainer.run_training(num_iterations=500)

    print("--- 4. Evaluation ---")
    metrics = trainer.regreccion_test_accuracy(X, y, num_features=1)
    print(f"R2 Score: {metrics['R2']:.4f}")
    
    # Optional: Predict
    test_val = np.array([[1.5]]) # Roughly PI/2
    pred = trainer.predict(test_val)
    print(f"Sin(1.5) Real: {np.sin(1.5):.4f}, Pred: {pred[0,0]:.4f}")

    # --- 5. Visualization (Optional) ---
    print("--- 5. Plotting Results ---")
    
    # Predict on the whole range to see the curve
    predictions = trainer.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=1, label='True Data (Sin)', alpha=0.5)
    plt.plot(X, predictions, color='red', label='Neural Net Prediction')
    plt.title("HeteroSymNN: Sine Wave Regression")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simple_demo()