import sys
import os
import pytest
import numpy as np

# Ensure HeteroSymNN is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HeteroSymNN.config import settings
from HeteroSymNN.exceptions import (
    NetworkStructureError,
    LayerConfigurationError,
    ComputationalMethodValueError,
    DeviceSelectionError,
    ShapeMismatchError,
    WrapperError,
    TrainingError,
    LoadingError,
    SavingError,
    PathError
)
from HeteroSymNN.Core.Nets import LinearNet
from HeteroSymNN.API.wrappers import Wrapper

class TestNetworkExceptions:
    """
    Test suite for Network-related exceptions, ensuring invalid configurations
    are caught and handled gracefully.
    """
    
    def test_network_structure_error_few_layers(self):
        """Test that passing < 2 layers in nodes_structure raises NetworkStructureError."""
        with pytest.raises(NetworkStructureError, match="have at least 2"):
            LinearNet(nodes_structure=[4], activation_config=["relu"], num_training_iter=1)
            
    def test_network_structure_error_mismatched_activations(self):
        """Test that mismatch between nodes_structure length and activation_config length raises NetworkStructureError."""
        # 3 layers total -> expecting 2 activations, passing 1
        with pytest.raises(NetworkStructureError, match="but was set"):
            LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu"], num_training_iter=1)

    def test_layer_configuration_error_invalid_type(self):
        """Test that passing an invalid type (like an int) instead of a str/tuple for activation raises LayerConfigurationError."""
        with pytest.raises(LayerConfigurationError, match="Invalid activation configuration"):
            LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", 42], num_training_iter=1)

    def test_computational_method_value_error(self):
        """Test setting an unknown computational method."""
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        with pytest.raises(ComputationalMethodValueError, match="isn't GPU_CUDA, CPU_JIT or CPU_PYTHON"):
            model.set_backend("TPU_MAGIC")

    def test_device_selection_error(self):
        """Test changing device to an unknown hardware type."""
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        with pytest.raises(DeviceSelectionError, match="is not host or device"):
            model.to("TPU")

    def test_shape_mismatch_predict(self):
        """Test sending input with wrong feature dimensions to predict."""
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        bad_input = np.random.rand(10, 5).astype(np.float32) # Expects 4 features, sending 5
        with pytest.raises(ShapeMismatchError, match="none dimention matched|Shape of the input is incorrect"):
            model.predict(bad_input)


class TestWrapperExceptions:
    """
    Test suite for Wrapper/API level exceptions.
    """
    
    @pytest.fixture
    def basic_wrapper(self):
        model = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=1)
        return Wrapper(model, work_type="reg")

    def test_wrapper_invalid_model(self):
        """Test wrapping something that isn't a BaseNetwork subclass."""
        with pytest.raises(WrapperError, match="is not a subclass of BaseNetwork"):
            Wrapper(model="Not a network", work_type="reg")

    def test_wrapper_missing_work_type(self):
        """Test that missing or invalid work_type throws an error during evaluation."""
        model = LinearNet(nodes_structure=[4, 4, 1], activation_config=["relu", "linear"], num_training_iter=1)
        agent = Wrapper(model, work_type=None)
        with pytest.raises(WrapperError, match="Work type was not specified"):
            agent.test_accuracy(np.random.rand(10, 4), np.random.rand(10, 1))

    def test_shape_mismatch_different_samples(self, basic_wrapper):
        """Test load_training with X and y having different number of rows/samples."""
        X = np.random.rand(10, 4)
        y = np.random.rand(12, 1) # 12 samples instead of 10
        with pytest.raises(ShapeMismatchError, match="Sample count mismatch"):
            basic_wrapper.load_training(X, y)

    def test_shape_mismatch_wrong_y_features(self, basic_wrapper):
        """Test load_training with y having the wrong number of output columns."""
        X = np.random.rand(10, 4)
        y = np.random.rand(10, 2) # 2 columns, but network output is 1
        with pytest.raises(ShapeMismatchError, match="Target shape mismatch"):
            basic_wrapper.load_training(X, y)
            
    def test_training_error_run_without_load(self, basic_wrapper):
        """Test that run_training fails if data hasn't been loaded."""
        with pytest.raises(TrainingError, match="No Training data loaded"):
            basic_wrapper.run_training(num_iterations=1, batch_size=2)

    def test_invalid_validation_split(self, basic_wrapper):
        """Test validation_split outside the bounds of 0-1 using GridSearchManager."""
        from HeteroSymNN.API.wrappers import GridSearchManager
        with pytest.raises(WrapperError, match="validation_split value should be between 0 and 1"):
            GridSearchManager(basic_wrapper, param_grid={}, validation_split=1.5)


class TestRuntimeStateExceptions:
    """
    Test suite for Saving/Loading IO exceptions.
    """
    
    def test_path_error_on_load(self):
        """Test loading from a non-existent file path."""
        with pytest.raises(PathError, match="No file found"):
            LinearNet.load_model("this_path_does_not_exist.symnn")
            
    def test_loading_error_wrong_topology(self, tmp_path):
        """Test load_state into a network with different topology."""
        model_save = LinearNet(nodes_structure=[4, 8, 1], activation_config=["relu", "linear"], num_training_iter=1)
        model_save.predict(np.zeros((1, 4), dtype=np.float32)) # Initialize state to allow saving
        save_file = str(tmp_path / "test_model")
        model_save.save_model(save_file)
        
        # Different topology (missing the hidden layer)
        model_load = LinearNet(nodes_structure=[4, 1], activation_config=["linear"], num_training_iter=1)
        with pytest.raises(LoadingError, match="has a different number of layers"):
            model_load.load_state(save_file)
