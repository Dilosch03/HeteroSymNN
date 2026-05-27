import sys
sys.path.insert(0, r"C:\Users\dilos\Documents\GitHub\HeteroSymNN")

import numpy as np
import time
from HeteroSymNN.Core.Nets import LinearNet, HeteroLinearNet
from HeteroSymNN.API import Wrapper, GridSearchManager
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
    template_model = LinearNet(
        nodes_structure=[2, 16, 16, 1],
        activation_config=["relu", "relu", "sigmoid"],
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

    print("\n--- PART 2: Using the Best Model ---")
    # Let's test the best wrapper on a couple of manual examples
    test_points = np.array([[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
    expected_labels = np.array([[0], [0], [1], [1]])
    
    print("Testing manual XOR points:")
    predictions = best_wrapper.predict(test_points)
    
    for i in range(4):
        pred_val = predictions[i][0]
        # It's a binary classification, so if pred > 0.5 it's class 1
        pred_class = 1 if pred_val > 0.5 else 0
        expected = expected_labels[i][0]
        print(f"Input {test_points[i]} -> Pred: {pred_val:.4f} (Class {pred_class}) | Expected: {expected}")


if __name__ == "__main__":
    run_expert_demo()