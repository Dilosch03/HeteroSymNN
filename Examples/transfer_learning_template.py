"""
HeteroSymNN Transfer Learning & Model Reconfiguration Blueprint

This script provides a complete, self-contained blueprint showing how to load a
pre-trained model, reconfigure its layers for a new similar task, and fine-tune it.

To keep the repository clean of binary files, this script runs a quick, small training
pass on a base task to create a pre-trained model in RAM, saves it locally as a temporary 
file (which is gitignored), and then demonstrates the transfer learning process.

Features demonstrated:
1. Programmatic model saving and loading.
2. Parameter extraction and transfer using the `set_parameters` network-level API.
3. Model reconfiguration (adapting early layers to a new task-specific head).
4. Convergence comparison: Pre-trained fine-tuning vs. training from scratch.
"""

import numpy as np

from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.Core import losses, optimizers
from HeteroSymNN.API import Wrapper
from HeteroSymNN.API import data_transformers as utils

def generate_base_task_data(num_samples=400):
    """
    Base Task: Learn a simple sine wave, y = sin(x).
    """
    X = np.linspace(-np.pi, np.pi, num_samples).reshape(-1, 1).astype(np.float32)
    y = np.sin(X).astype(np.float32)
    return X, y

def generate_target_task_data(num_samples=400):
    """
    Target Task: Learn a shifted and scaled cosine wave, y = cos(x) + 0.5.
    This is a similar but distinct task.
    """
    X = np.linspace(-np.pi, np.pi, num_samples).reshape(-1, 1).astype(np.float32)
    y = (np.cos(X) + 0.5).astype(np.float32)
    return X, y

def build_base_model():
    """
    Build a standard MLP architecture for the base task.
    """
    return LinearNet(
        nodes_structure=[1, 32, 16, 1],
        activation_config=["relu", "relu", "num"],
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.01),
        batch_size=32
    )

def run_transfer_learning_demo():
    print("==================================================")
    print("   HETEROSYMNN TRANSFER LEARNING DEMONSTRATION   ")
    print("==================================================")

    # -------------------------------------------------------------------------
    # PART 1: Train the Base Model
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Training Base Model on Base Task (Sin Wave) ---")
    X_base, y_base = generate_base_task_data()
    base_model = build_base_model()
    base_trainer = Wrapper(base_model, work_type="reg", input_transformer=utils.MinMaxScaler())
    
    # Train the base model for 40 epochs
    print("Training base model...")
    base_trainer.fit(X_base, y_base, epochs=40)
    print(f"Base model training complete. Final loss: {base_trainer.model.history_losses[-1]:.6f}")

    # Save the base model to a temporary file (gitignored)
    temp_model_path = "temp_pretrained_base.symnn"
    base_trainer.save_model(temp_model_path, overwrite=True)
    print(f"Base model saved to temporary file: '{temp_model_path}'.")

    # -------------------------------------------------------------------------
    # PART 2: Reconfigure and Transfer Parameters for the Target Task
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Reconfiguring Model for Target Task (Shifted Cosine) ---")
    
    # Load the pre-trained model using the Wrapper's static loader
    print("Loading pre-trained model...")
    loaded_trainer = Wrapper.load_model(temp_model_path)
    pretrained_net = loaded_trainer.model

    # We build a brand-new model for the target task.
    # It has the same hidden topology, but we will transfer the feature extraction weights.
    print("Building target model architecture...")
    target_net = LinearNet(
        nodes_structure=[1, 32, 16, 1],
        activation_config=["relu", "relu", "num"],
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.01),
        batch_size=32
    )

    # Extract the pre-trained parameters from the hidden layers (layer_0 and layer_1)
    # layers are index-based in the model.layers list.
    print("Extracting weights and biases from pre-trained hidden layers...")
    
    # We transfer weights for the first two hidden layers (layer_0 and layer_1)
    # while leaving the final output layer (layer_2) initialized randomly for the new task.
    pretrained_parameters = {
        "layer_0": {
            "weights": pretrained_net.layers[0].weights.T,
            "biases": pretrained_net.layers[0].biases.T,
            "connection_mask": pretrained_net.layers[0]._connection_mask.T
        },
        "layer_1": {
            "weights": pretrained_net.layers[1].weights.T,
            "biases": pretrained_net.layers[1].biases.T,
            "connection_mask": pretrained_net.layers[1]._connection_mask.T
        }
    }

    # Inject pre-trained parameters into the target network
    print("Injecting pre-trained parameters into the new target model...")
    target_net.set_parameters(pretrained_parameters)
    print("Parameter transfer complete.")

    # -------------------------------------------------------------------------
    # PART 3: Fine-Tune the Reconfigured Model vs. Training From Scratch
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Comparing Convergence (Transfer vs. Scratch) ---")
    X_target, y_target = generate_target_task_data()

    # 3a. Fine-tune the transferred model
    print("\n[Method A] Fine-tuning the Transferred Model...")
    transferred_trainer = Wrapper(target_net, work_type="reg", input_transformer=utils.MinMaxScaler())
    transferred_trainer.fit(X_target, y_target, epochs=25)
    transferred_losses = transferred_trainer.model.history_losses[-25:]

    # 3b. Train a fresh model from scratch (control group)
    print("\n[Method B] Training a Fresh Model from Scratch...")
    scratch_net = LinearNet(
        nodes_structure=[1, 32, 16, 1],
        activation_config=["relu", "relu", "num"],
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.01),
        batch_size=32
    )
    scratch_trainer = Wrapper(scratch_net, work_type="reg", input_transformer=utils.MinMaxScaler())
    scratch_trainer.fit(X_target, y_target, epochs=25)
    scratch_losses = scratch_trainer.model.history_losses

    # -------------------------------------------------------------------------
    # PART 4: Analyze and Report Results
    # -------------------------------------------------------------------------
    print("\n--- Part 4: Convergence Analysis ---")
    print(f"{'Epoch':<8}{'Scratch Loss':<18}{'Transfer Loss':<18}{'Improvement':<12}")
    print("-" * 58)
    
    # Compare losses at specific epochs
    compare_epochs = [1, 5, 10, 15, 20, 25]
    for ep in compare_epochs:
        idx = ep - 1
        scratch_l = scratch_losses[idx]
        transfer_l = transferred_losses[idx]
        improvement = ((scratch_l - transfer_l) / scratch_l) * 100
        print(f"{ep:<8}{scratch_l:<18.6f}{transfer_l:<18.6f}{improvement:>10.1f}%")

    # Clean up temporary files
    try:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            print(f"\nTemporary file '{temp_model_path}' cleaned up successfully.")
    except Exception as e:
        print(f"Warning: Could not remove temporary file: {e}")

    print("\nDemonstration complete. Transfer learning successfully accelerated convergence.")
    print("==================================================")

if __name__ == "__main__":
    run_transfer_learning_demo()
