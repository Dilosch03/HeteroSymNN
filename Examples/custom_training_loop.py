"""
HeteroSymNN Low-Level JIT Primitives Training Blueprint

This script provides a highly polished blueprint showing how to bypass the
high-level Wrapper agent and write a custom training loop using JIT primitives directly.

It demonstrates:
1. Low-level data preparation and device casting.
2. Direct execution of forward, backward, and parameter updates.
3. Implementing custom training logic: validation splits and early stopping.
4. Telemetry tracking of both training and validation losses.
"""

import numpy as np

from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.Core import losses, optimizers

def generate_dataset(num_samples=800):
    """
    Generate synthetic quadratic data: y = x^2 + noise.
    """
    X = np.linspace(-2.0, 2.0, num_samples).reshape(-1, 1).astype(np.float32)
    y = (X**2 + np.random.normal(0, 0.15, size=X.shape)).astype(np.float32)
    return X, y

def train_custom_loop():
    print("==================================================")
    print("   HETEROSYMNN LOW-LEVEL CUSTOM TRAINING LOOP     ")
    print("==================================================")

    # -------------------------------------------------------------------------
    # PART 1: Data Preparation & Validation Split
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Preparing Datasets & Splits ---")
    X, y = generate_dataset(num_samples=1000)
    
    # 80/20 Train/Validation Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train samples:      {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # -------------------------------------------------------------------------
    # PART 2: Model Instantiation
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Instantiating Network with JIT Backend ---")
    # We define a network topology with a batch size of 32
    batch_size = 32
    model = LinearNet(
        nodes_structure=[1, 32, 16, 1],
        activation_config=["relu", "relu", "num"],
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.01),
        batch_size=batch_size
    )

    # Compile and load the network onto the default hardware device (GPU if available, else CPU JIT)
    model.to("device")
    print("JIT kernels compiled and loaded to the active device.")

    # -------------------------------------------------------------------------
    # PART 3: Low-Level Training Loop with Early Stopping
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Running Custom JIT Training Loop ---")
    epochs = 100
    patience = 8
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Convert validation set to column-major format and cast to active device
    # HeteroSymNN expects features as rows, samples as columns (features, samples)
    x_val_dev, y_val_dev = model.cast_arrays(X_val.T, y_val.T)

    # Transpose training data for batch slicing (features, samples)
    train_features = X_train.T
    train_targets = y_train.T
    num_train_samples = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        # Shuffle indices at the start of each epoch
        shuffled_indices = np.random.permutation(num_train_samples)
        epoch_loss_accum = 0.0
        batch_count = 0

        # Mini-batch training
        for start_idx in range(0, num_train_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_train_samples)
            batch_indices = shuffled_indices[start_idx:end_idx]
            
            # Slice batch (numpy operations on CPU)
            x_batch_cpu = train_features[:, batch_indices]
            y_batch_cpu = train_targets[:, batch_indices]
            
            # 1. Cast batch data to active computational device (RAM/VRAM)
            x_batch, y_batch = model.cast_arrays(x_batch_cpu, y_batch_cpu)
            
            # 2. Run Forward Pass
            y_pred = model.forward(x_batch)
            
            # 3. Calculate Loss & Error (gradient of loss w.r.t predictions)
            loss_val = model.loss_function.forward(y_pred, y_batch)
            error = model.loss_function.backward(y_pred, y_batch)
            
            # Accumulate loss (converting JIT array back to python float)
            epoch_loss_accum += float(model._ASNUMPY(loss_val))
            batch_count += 1
            
            # 4. Run Backward Pass (backpropagates gradients through JIT layers)
            model.backward(error)
            
            # 5. Update Parameters (optimizer updates weights/biases on device)
            model.update_params()

        # Calculate average training loss for this epoch
        train_loss = epoch_loss_accum / batch_count

        # ---------------------------------------------------------------------
        # Validation Pass (Forward Pass only - no backward or updates)
        # ---------------------------------------------------------------------
        y_val_pred = model.forward(x_val_dev)
        val_loss_val = model.loss_function.forward(y_val_pred, y_val_dev)
        val_loss = float(model._ASNUMPY(val_loss_val))

        # Print telemetry details
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # (Optional) Save the best model state here
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch}. Validation loss did not improve for {patience} epochs.")
                print(f"Best Validation Loss achieved: {best_val_loss:.6f}")
                break

    print("\nCustom training loop execution complete.")
    print("==================================================")

if __name__ == "__main__":
    train_custom_loop()
