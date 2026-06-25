"""
HeteroSymNN Programmatic Training Boilerplate

This script provides a production-grade template for training custom architectures
programmatically using the high-level Wrapper agent.

Features demonstrated:
1. Programmatic model instantiation and wrapper setup.
2. Telemetry logging of epoch losses to a CSV file.
3. Periodic checkpoint saving.
4. Graceful interrupt handling (Ctrl+C / SIGINT) to save an emergency checkpoint.
5. Automatic hardware-accelerated training (CPU JIT / GPU CUDA) managed by the framework.
"""

import numpy as np
import csv

from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.Core import losses, optimizers
from HeteroSymNN.API import Wrapper
from HeteroSymNN.API import data_transformers as utils

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic 1D sine wave data with noise for regression.
    """
    X = np.linspace(-np.pi, np.pi, num_samples).reshape(-1, 1).astype(np.float32)
    # Target is sin(x) with some Gaussian noise
    y = (np.sin(X) + np.random.normal(0, 0.1, size=X.shape)).astype(np.float32)
    return X, y

def save_telemetry_log(history_losses, filename="training_log.csv"):
    """
    Save the list of training losses to a CSV file for downstream visualization.
    """
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss"])
            for epoch, loss in enumerate(history_losses, start=1):
                writer.writerow([epoch, loss])
        print(f"Telemetry log saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving telemetry log: {e}")

def run_production_training():
    print("--- 1. Generating Training Data ---")
    X, y = generate_synthetic_data(num_samples=1200)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} input features.")

    print("\n--- 2. Building the Network Architecture ---")
    # We define a multi-layer perceptron topology programmatically.
    # Hidden layers use JIT-compiled 'relu' activations, output layer uses linear 'num'.
    model = LinearNet(
        nodes_structure=[1, 32, 32, 1],
        activation_config=["relu", "relu", "num"],
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.01),
        batch_size=64
    )

    # Wrap the network. The Wrapper handles normalization, device management,
    # and provides convenient training/saving abstractions.
    trainer = Wrapper(
        model=model,
        work_type="reg",
        input_transformer=utils.MinMaxScaler()
    )

    # Load and prepare training data (computes normalization bounds)
    trainer.load_training(X, y)

    # Training configuration
    total_epochs = 100
    checkpoint_interval = 20
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("\n--- 3. Running Resilient Training Loop ---")
    print(f"Starting training for {total_epochs} epochs.")
    print("You can press 'Ctrl+C' at any time to interrupt training and save progress.")

    # We run training in increments to allow periodic checkpoint saving.
    current_epoch = 0
    try:
        while current_epoch < total_epochs:
            # Determine epochs to run in this interval
            epochs_to_run = min(checkpoint_interval, total_epochs - current_epoch)
            
            # Run training slice
            trainer.run_training(num_iterations=epochs_to_run)
            current_epoch += epochs_to_run
            
            # Print progress and current loss
            current_loss = trainer.model.history_losses[-1]
            print(f"Epoch {current_epoch}/{total_epochs} - Loss: {current_loss:.6f}")
            
            # Save periodic checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_epoch_{current_epoch}.symnn")
            trainer.save_model(checkpoint_path, overwrite=True)
            print(f"Saved periodic checkpoint to '{checkpoint_path}'.")

        print("\nTraining completed successfully.")

    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user (SIGINT / Ctrl+C).")
        # Save emergency checkpoint immediately
        emergency_path = "checkpoint_interrupted.symnn"
        trainer.save_model(emergency_path, overwrite=True)
        print(f"Emergency checkpoint saved successfully to '{emergency_path}'.")
        
        # Save telemetry up to the point of interruption
        save_telemetry_log(trainer.model.history_losses)
        print("Exiting cleanly.")
        sys.exit(0)

    print("\n--- 4. Saving Final Outputs ---")
    # Save final trained model
    final_model_path = "final_trained_model.symnn"
    trainer.save_model(final_model_path, overwrite=True)
    print(f"Final model saved to '{final_model_path}'.")

    # Save complete telemetry log
    save_telemetry_log(trainer.model.history_losses)

if __name__ == "__main__":
    run_production_training()
