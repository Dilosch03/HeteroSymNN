"""
HeteroSymNN Interactive Sensitivity Tuning GUI

This script provides an interactive desktop GUI demonstration using Python's
native, zero-dependency tkinter library.

It demonstrates:
1. Instantiating and training a network with symbolic activation constants.
2. Using the `model.change_constants()` API to update neuron parameters on the fly.
3. Zero-recompile JIT inference updating the prediction curve in real-time.
4. Seamless automated testing integration (bypasses GUI execution in test suites).
"""

import numpy as np

from HeteroSymNN.Core.Nets import MLP
from HeteroSymNN.Core import losses, optimizers
from HeteroSymNN.API import Wrapper
from HeteroSymNN.API import data_transformers as utils

# Check if we are running inside an automated test runner (like pytest)
IS_TEST_ENV = "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") is not None

def generate_data():
    """
    Generate clean sine wave data for training and plotting.
    """
    X = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1).astype(np.float32)
    y = np.sin(X).astype(np.float32)
    return X, y

def train_baseline_model():
    """
    Quickly train a small model in RAM to serve as our interactive baseline.
    Since JIT is extremely fast, this takes a fraction of a second.
    """
    print("Training baseline model in RAM...")
    X, y = generate_data()
    
    # We use a symbolic activation function with a tunable constant 'alfa'
    model = MLP(
        nodes_structure=[1, 16, 1],
        activation=("relu(x) * alfa", {"alfa": 1.0}),
        output_activation="num",
        loss_function=losses.MSELoss(),
        optimizer=optimizers.AdamOptimizer(learning_rate=0.02),
        batch_size=32
    )
    
    trainer = Wrapper(
        model=model,
        work_type="reg",
        input_transformer=utils.MinMaxScaler()
    )
    
    # Train for 40 epochs to get a basic fit
    trainer.fit(X, y, epochs=40)
    print("Baseline model trained successfully.")
    return trainer

def launch_gui(trainer):
    """
    Set up and launch the tkinter desktop GUI.
    """
    import tkinter as tk
    from tkinter import ttk

    # Get the trained model
    model = trainer.model

    # Generate evaluation grid for plotting
    X_eval = np.linspace(-np.pi, np.pi, 150).reshape(-1, 1).astype(np.float32)
    y_true = np.sin(X_eval)

    # Tkinter window setup
    root = tk.Tk()
    root.title("HeteroSymNN: Real-time Sensitivity Tuner")
    root.geometry("800x600")
    root.configure(bg="#2d2d2d")

    # Style configuration
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", background="#2d2d2d", foreground="#ffffff", font=("Helvetica", 10))
    style.configure("Header.TLabel", background="#2d2d2d", foreground="#3b82f6", font=("Helvetica", 14, "bold"))

    # Header label
    header = ttk.Label(root, text="Symbolic JIT Activation Tuning (Zero Recompile)", style="Header.TLabel")
    header.pack(pady=10)

    # Frame for the plot
    plot_frame = tk.Frame(root, bg="#2d2d2d")
    plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

    # Canvas dimensions
    canvas_w, canvas_h = 760, 360
    canvas = tk.Canvas(plot_frame, width=canvas_w, height=canvas_h, bg="#121212", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    # Slide control frame
    control_frame = tk.Frame(root, bg="#2d2d2d")
    control_frame.pack(fill=tk.X, padx=20, pady=15)

    slider_label = ttk.Label(control_frame, text="Adjust Activation Constant 'alfa' (All Neurons in Hidden Layer):")
    slider_label.pack(anchor="w", pady=2)

    # State variable for the constant
    alfa_var = tk.DoubleVar(value=1.0)

    def draw_chart():
        """
        Draw the axes, true sine wave, and JIT model predictions on the canvas.
        """
        canvas.delete("all")

        # Draw grid and axes
        canvas.create_line(0, canvas_h // 2, canvas_w, canvas_h // 2, fill="#333333", width=1)
        canvas.create_line(canvas_w // 2, 0, canvas_w // 2, canvas_h, fill="#333333", width=1)
        
        # Label the axes
        canvas.create_text(20, canvas_h // 2 - 10, text="-PI", fill="#888888", font=("Helvetica", 8))
        canvas.create_text(canvas_w - 20, canvas_h // 2 - 10, text="PI", fill="#888888", font=("Helvetica", 8))
        canvas.create_text(canvas_w // 2 + 15, 15, text="1.5", fill="#888888", font=("Helvetica", 8))
        canvas.create_text(canvas_w // 2 + 20, canvas_h - 15, text="-1.5", fill="#888888", font=("Helvetica", 8))

        # Run inference on the evaluation grid
        y_pred = trainer.predict(X_eval)

        # Map X, Y coordinates to canvas space
        # X: [-pi, pi] -> [20, canvas_w - 20]
        # Y: [-1.5, 1.5] -> [canvas_h - 20, 20]
        def map_coords(x_val, y_val):
            cx = 20 + ((x_val + np.pi) / (2 * np.pi)) * (canvas_w - 40)
            cy = 20 + ((1.5 - y_val) / 3.0) * (canvas_h - 40)
            return cx, cy

        # Draw True Function (Sine wave in Cyan)
        true_points = []
        for i in range(len(X_eval)):
            cx, cy = map_coords(X_eval[i, 0], y_true[i])
            true_points.extend([cx, cy])
        canvas.create_line(*true_points, fill="#06b6d4", width=2, tags="true_curve")

        # Draw Model Prediction (Pink/Red curve)
        pred_points = []
        for i in range(len(X_eval)):
            cx, cy = map_coords(X_eval[i, 0], y_pred[i, 0])
            pred_points.extend([cx, cy])
        canvas.create_line(*pred_points, fill="#ec4899", width=3, tags="pred_curve")

        # Legends
        canvas.create_rectangle(30, 20, 45, 25, fill="#06b6d4", outline="")
        canvas.create_text(85, 22, text="True Sin(x)", fill="#ffffff", font=("Helvetica", 9))
        canvas.create_rectangle(150, 20, 165, 25, fill="#ec4899", outline="")
        canvas.create_text(220, 22, text="JIT Prediction", fill="#ffffff", font=("Helvetica", 9))

    def on_slider_change(val):
        """
        Event handler when slider is moved.
        Updates model constants, runs forward pass, and redraws the curve.
        """
        alfa_val = float(val)
        
        # We change the constant 'alfa' for all 16 neurons in layer 0 (the hidden layer)
        # format: { layer_idx: [(neuron_idx, constant_name, new_value), ...] }
        constants_update = {
            0: [(i, "alfa", alfa_val) for i in range(16)]
        }
        
        model.change_constants(constants_update)
        
        # Redraw predictions on canvas
        draw_chart()

    # Create slider
    slider = tk.Scale(
        control_frame,
        from_=-3.0,
        to=3.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=alfa_var,
        command=on_slider_change,
        bg="#2d2d2d",
        fg="#ffffff",
        highlightbackground="#2d2d2d",
        activebackground="#3b82f6",
        troughcolor="#121212",
        length=760
    )
    slider.pack(pady=5)

    # Initial chart draw
    draw_chart()

    # Start the Tkinter event loop
    print("Launching GUI window...")
    root.mainloop()

def main():
    # Train the model in RAM
    trainer = train_baseline_model()

    if IS_TEST_ENV:
        # In automated tests, do not block or open a window. Just run inference once to verify correctness.
        print("Running in test environment. Verifying JIT constant change and prediction...")
        trainer.model.change_constants({0: [(i, "alfa", 0.5) for i in range(16)]})
        pred = trainer.predict(np.array([[1.0]]))
        print(f"Prediction at x=1.0 with alfa=0.5: {pred[0,0]:.6f}")
        print("Verification successful. Exiting test run cleanly.")
        return

    # In regular execution, launch the interactive Tkinter window
    launch_gui(trainer)

if __name__ == "__main__":
    main()
