import os
import io
import numpy as np
import math as mth
import itertools as iter
from typing import Literal,Union,Optional
import datetime
import warnings
import inspect
import time
import json
import zipfile


from ..Backend import hardware as HW
from ..types import LayerConstruction,NodeConfig
from ..Core.Nets.base_classes import BaseNetwork
from ..Core.layers import BaseLayer
from ..Core import losses, optimizers,initializers
from . import registries,utilities
from ..exceptions import PathError,PathWarning,ShapeMismatchError,ShapeWarning,LoadingError,TrainingError,LoadingWarning,WrapperError,SavingError
from ..config import settings

class Wrapper():
    """
    A high-level wrapper for managing the lifecycle of a :class:`~HeteroSymNN.Core.Nets.BaseNetwork` or subclass of :class:`~HeteroSymNN.Core.Nets.BaseNetwork`

    This class simplifies common tasks such as:
    
    *   Data loading and scaling (if :obj:`HeteroSymNN.API.DataTransformer` is pass).
    *   Training execution and loss tracking.
    *   Model evaluation (Classification metrics or Regression metrics).
    *   Saving and loading the full model state (architecture, weights, optimizer state).

    Parameters
    ----------
    model : :class:`~HeteroSymNN.Core.Nets.BaseNetwork` | subclass of :class:`~HeteroSymNN.Core.Nets.BaseNetwork` or None
        The neural network instance to wrap. Can be ``None`` if loading a model from disk later.
    work_type : Literal["class", "reg"]
        Can be ``None`` if loading a model from disk later.
        The type of problem the model solves:
        
        *   ``"class"``: Classification (calculates Accuracy, F1, etc.).
        *   ``"reg"``: Regression (calculates MSE, R2, etc.).

    _input_transformer : 'obj:`HeteroSymNN.API.DataTransformer`, optional
        The scaling method that is going to be use to scale the input data. Defaults to ``None`` and doesn't scale the data.
    _output_transformer : 'obj:`HeteroSymNN.API.DataTransformer`, optional
        The descaling method that is going to be use to scale the input data. Defaults to ``None`` and doesn't descale the data.
    """
    def __init__(self, model: BaseNetwork,work_type:Literal["class","reg"],input_transformer: Optional[utilities.DataTransformer] = None, output_transformer: Optional[utilities.DataTransformer] = None):
        self._model:BaseNetwork = None
        self.training_data = None
        self.training_data_norm = None
        self._input_transformer = input_transformer
        self._output_transformer = output_transformer
        self.work_type = work_type
        self._loaded_train_data = False
        self.model_name = model.__class__.__name__

        self.model:BaseNetwork = model

    @property
    def model(self)->BaseNetwork:
        """
        The underlying neural network instance.
        
        When setting this property, the wrapper validates that the new model's input/output dimensions 
        match any previously loaded training data.
        """
        return self._model
    
    @model.setter
    def model(self, new_model:BaseNetwork):
        self._model:BaseNetwork = new_model

        if ((new_model != None) and (self._loaded_train_data)):
            X_norm, Y_norm = self.training_data_norm
            
            expected_x = new_model.layers[0].num_inputs
            if X_norm.ndim == 2 and X_norm.shape[1] != expected_x:
                if (settings.warning_level=="error"):
                    raise ShapeMismatchError(f"The model new is expecting {expected_x} features, but the loaded data has {X_norm.shape[1]} features.")
                elif(settings.warning_level == "warn"):
                    warnings.warn(f"The model new is expecting {expected_x} features, but the loaded data has {X_norm.shape[1]} features.",ShapeMismatchError,stacklevel=2)
            expected_y = new_model.layers[-1].num_nodes
            if Y_norm.ndim == 2 and Y_norm.shape[1] != expected_y:
                if (settings.warning_level=="error"):
                    raise ShapeMismatchError(f"The model new is expecting {expected_y} features, but the loaded data has {Y_norm.shape[1]} targets.")
                elif(settings.warning_level == "warn"):
                    warnings.warn(f"The model new is expecting {expected_y} features, but the loaded data has {Y_norm.shape[1]} targets.",ShapeMismatchError,stacklevel=2)
            
    @property
    def input_transformer(self):
        return self._input_transformer

    @property
    def output_transformer(self):
        return self._output_transformer

    def fit(self, training_data: list, expected_results: list, epochs: int = None,training_mode: Literal["batch", "mini-batch", "stochastic"] = None, batch_size: int = None)->list[float]:
        """
        Method like Scikit Learn for training the model.

        Parameters
        ----------
        training_data : list or np.ndarray
            Input features (X). Shape should be (n_samples, n_features).
        expected_results : list or np.ndarray
            Target labels/values (Y). Shape should be (n_samples, n_outputs).
        epochs : int, optional
            Number of epochs to train. If None, uses the model's configured default.
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training strategy. If None, uses the model's configured default.
        batch_size : int, optional
            Size of the batch for "mini-batch" mode. If not pass, uses the model's configured default.

        Returns
        -------
        list[float]
            A list of loss values recorded during training.
        
        Raises
        ------
        ValueError
            If dimensions do not match the model's expected input/output size.
        """
        self.load_training(training_data, expected_results)
        return self.run_training(epochs, training_mode, batch_size)

    def load_training(self, training_data: list, expected_results: list)->None:
        """
        Loads and prepares training data.

        This method performs the following steps:
        
        1.  Converts inputs to NumPy arrays.
        2.  Checks dimensions against the model architecture (if a model is set).
        3.  Calculates normalization statistics (min/max) if normalization is enabled.
        4.  Stores the normalized data for training.

        Parameters
        ----------
        training_data : list or np.ndarray
            Input features (X). Shape should be (n_samples, n_features).
        expected_results : list or np.ndarray
            Target labels/values (Y). Shape should be (n_samples, n_outputs).
        
        Raises
        ------
        ValueError
            If dimensions do not match the model's expected input/output size.
        """
        self._loaded_train_data = True
        X_raw = np.array(training_data)
        Y_raw = np.array(expected_results)
        
        if X_raw.ndim == 1:
            if (settings.warning_level=="error"):
                raise ShapeMismatchError("The shape of the inputs are 1D.")
            elif(settings.warning_level == "warn"):
                warnings.warn("The shape of the inputs are 1D. Resheaping it to (num features, 1).",ShapeWarning,stacklevel=2)
                X_raw = X_raw.reshape(-1, 1)
            
        if Y_raw.ndim == 1:
            if (settings.warning_level=="error"):
                raise ShapeMismatchError("The shape of the outputs are 1D.")
            elif(settings.warning_level == "warn"):
                warnings.warn("The shape of the outputs are 1D. Resheaping it to (num targets, 1).",ShapeWarning,stacklevel=2)
                Y_raw = Y_raw.reshape(-1, 1)
        
        if (self._model != None):
            expected_x_features = self.model.layers[0].num_inputs
            if X_raw.ndim == 2 and X_raw.shape[1] != expected_x_features:
                if X_raw.shape[0] == expected_x_features:
                    if (settings.warning_level=="error"):
                        raise ShapeMismatchError(f"The input data looks to be in the format (features, samples).")
                    elif(settings.warning_level == "warn"):
                        warnings.warn("The input data looks to be in the format (features, samples).Transposing to (samples, features).",ShapeWarning,stacklevel=2)
                        X_raw = X_raw.T
                else:
                    raise ShapeMismatchError(f"The shape of the inputs are {X_raw.shape}, but the model expects {expected_x_features} features.")

            expected_y_features = self.model.layers[-1].num_nodes      
            if Y_raw.ndim == 2 and Y_raw.shape[1] != expected_y_features:
                if Y_raw.shape[0] == expected_y_features:
                    if (settings.warning_level=="error"):
                        raise ShapeMismatchError("The output data looks to be in the format (outputs, samples).")
                    elif(settings.warning_level == "warn"):
                        warnings.warn("The output data looks to be in the format (outputs, samples).Transposing to (samples, outputs).",ShapeWarning,stacklevel=2)
                        Y_raw = Y_raw.T 
                else:
                    raise ShapeMismatchError(f"The shape of the outputs are {Y_raw.shape}, but the model expects {expected_y_features} outputs.")
            
        self.training_data = (X_raw, Y_raw)
        X_norm = X_raw

        if (self._input_transformer is not None):
            X_norm = self._input_transformer.fit_transform(X_raw)

        Y_norm = Y_raw
        if (self._output_transformer is not None):
            Y_norm = self._output_transformer.fit_transform(Y_raw)
            
        
        self.training_data_norm = (X_norm, Y_norm)

    def run_training(self, num_iterations:int = None, training_mode: Literal["batch", "mini-batch", "stochastic"] = None, batch_size: int = None)->list[float]:
        """
        Executes the training loop on the loaded data.

        Parameters
        ----------
        num_iterations : int, optional
            Number of epochs to train. If None, uses the model's configured default.
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training strategy. If None, uses the model's configured default.
        batch_size : int, optional
            Size of the batch for "mini-batch" mode. If not pass, uses the model's configured default.

        Returns
        -------
        list[float]
            A list of loss values recorded during training.
        """
        if self.training_data is None:
            raise TrainingError("No Training data loaded. Use load_training() first.")
        
        losses = self.model.train(
            self.training_data_norm[0],
            self.training_data_norm[1],
            num_iterations=num_iterations,
            training_mode=training_mode,
            batch_size=batch_size
        )
        return losses
    
    def predict(self, data: list)->np.ndarray:
        """
        Generates predictions for new data.

        Handles input normalization and output denormalization automatically if configured.

        Parameters
        ----------
        data : list or np.ndarray
            Input features to predict.

        Returns
        -------
        np.ndarray
            Predicted values (denormalized if ``normalize_outputs=True``).
        """
        X_raw = np.array(data)
        
        X_norm = X_raw
        if (self._input_transformer is not None):
            None_keys = []
            data_trans_config = self._input_transformer.get_config()
            for key in data_trans_config.keys():
                if data_trans_config[key] is None:
                    None_keys.append(key)

            if not(self._input_transformer.fitted):
                raise WrapperError("A DataTransform object was pass but it was never fitted.")
            
            if (len(None_keys) > 0):
                raise WrapperError(f"Input transformer has {",".join(None_keys)} with None values even though it was fitted.")
                
            X_norm = self._input_transformer.transform(X_raw)
            
            
        Y_pred_norm = self.model.predict(X_norm)
        
        Y_denorm = Y_pred_norm
        if (self._output_transformer is not None):
            None_keys = []
            data_trans_config = self._output_transformer.get_config()
            for key in data_trans_config.keys():
                if data_trans_config[key] is None:
                    None_keys.append(key)

            if not(self._output_transformer.fitted):
                raise WrapperError("A DataTransform object was pass but it was never fitted.")
            
            if (len(None_keys) > 0):
                raise WrapperError(f"Input transformer has {",".join(None_keys)} with None values even though it was fitted.")
                
            Y_denorm = self._output_transformer.transform(Y_pred_norm)
        
        return Y_denorm

    def test_accuracy(self,test_data:list,expected_results:list,threshold:float = 0.5)->Union[dict[str, float], tuple[dict[str, float], dict[str, int]]]:
        """
        Evaluates the model on a test dataset.

        Parameters
        ----------
        test_data : list
            Input features for testing.
        expected_results : list
            Ground truth values.
        threshold : float, optional
            Threshold for binary classification. Default is 0.5.

        Returns
        -------
        dict or tuple
            Evaluation metrics depending on ``work_type``.
        """
        results = {}
        if (self.work_type == "reg"):
            results = self.regreccion_test_accuracy(test_data,expected_results)
        elif(self.work_type == "class"):
            results = self.classification_test_accuracy(test_data,expected_results,threshold)
        else:
            raise WrapperError("Work type was not specified.")
        
        return results

    def classification_test_accuracy(self, test_data: list, expected_results: list,threshold:float = 0.5)->tuple[dict[str, float], dict[str, int]]:
        """
        Calculates classification metrics (Accuracy, Precision, Recall/TPR, F1 Score).

        Note: Currently assumes binary classification or multi-label, default threshold is 0.5.

        Returns
        -------
        tuple[dict[str, float], dict[str, int]]
            A tuple containing:
            
            1.  Dictionary of metrics (Acur, Press, TPR, F1).
            2.  Dictionary of raw counts (TP, TN, FP, FN).
        """
        predictions_raw = self.predict(test_data)
        
        predictions = (predictions_raw > threshold).astype(int).flatten()
        
        expected_results = list(expected_results)
        results_compare = {"correc_pos":0, "correct_neg":0, "false_pos":0, "false_neg":0}

        for i in range(len(expected_results)):
            is_equal = (predictions[i] == expected_results[i])
            if(is_equal):
                if(predictions[i]):
                    results_compare["correc_pos"] += 1
                else:
                    results_compare["correct_neg"] += 1
            else:
                if(predictions[i]):
                    results_compare["false_pos"] += 1
                else:
                    results_compare["false_neg"] += 1
        
        evals = {"Acur":-1, "Press":-1, "TPR":-1, "F1":-1}
        correct = results_compare["correc_pos"] + results_compare["correct_neg"]
        total = len(expected_results)
        
        try:
            evals["Acur"] = correct / total
        except ZeroDivisionError: pass
        try:
            evals["Press"] = results_compare["correc_pos"] / (results_compare["correc_pos"] + results_compare["false_pos"])
        except ZeroDivisionError: pass
        try:
            evals["TPR"] = results_compare["correc_pos"] / (results_compare["correc_pos"] + results_compare["false_neg"])
        except ZeroDivisionError: pass
        try:
            evals["F1"] = 2 * evals["Press"] * evals["TPR"] / (evals["Press"] + evals["TPR"])
        except ZeroDivisionError: pass

        return (evals, results_compare)
    
    def regreccion_test_accuracy(self, test_data: list, expected_results: list)->dict[str, float]:
        """
        Calculates regression metrics (R2, MSE, RMSE, MAE, MAPE, AIC, BIC).

        Returns
        -------
        dict[str, float]
            Dictionary containing the calculated metrics.
        """
        predictions_raw = self.predict(test_data)
        
        if predictions_raw.ndim == 2 and predictions_raw.shape[1] == 1:
            results = predictions_raw.flatten()
        else:
            results = predictions_raw
            
        expected_results = np.array(expected_results).flatten()
        
        if (results.shape != expected_results.shape):
            raise ShapeMismatchError(f"The expected results are not in the expected shape, Resived {expected_results.shape} and expected {results.shape}.")

        mean = np.mean(expected_results)
        rss = 0
        ssr = 0
        mae = 0
        mape = 0
        
        for i in range(len(results)):
            rss += (expected_results[i] - results[i])**2
            ssr += (results[i] - mean)**2
            mae += abs(expected_results[i] - results[i]) 
            try:
                mape += abs((expected_results[i] - results[i]) / expected_results[i])
            except ZeroDivisionError: pass

        if (np.isnan(mape)):
            mape = 0
        
        tss = rss + ssr
        n = len(expected_results)
        k = self.model.layers[0].num_inputs
        
        evaluations = {"TSS":tss, "RSS":rss, "SSR":ssr, "R2":np.nan, "MSE":np.nan, "RMSE":np.nan,
                       "MAPE":np.nan, "MAE":np.nan, "AIC":np.nan, "BIC":np.nan,
                       "TIME SERIES R2":np.nan, "APC":np.nan}
        try:
            evaluations["R2"] = 1 - (rss / tss)
        except ZeroDivisionError: pass
        try:
            evaluations["MSE"] = rss / n
        except ZeroDivisionError: pass
        try:
            evaluations["MAPE"] = mape / n
        except ZeroDivisionError: pass
        try:
            evaluations["MAE"] = mae / n
        except ZeroDivisionError: pass
        try:
            evaluations["AIC"] = -2 * mth.log(rss / n) + 2 * (k)
        except (ValueError, ZeroDivisionError): pass
        try:
            evaluations["BIC"] = (k) * mth.log(n) - n * mth.log(rss / n)
        except (ValueError, ZeroDivisionError): pass
        try:
            evaluations["APC"] = mth.sqrt(rss) + 2 * (k) / n
        except (ValueError, ZeroDivisionError): pass
        try:
            evaluations["TIME SERIES R2"] = 1 - ((1 - evaluations["R2"]) * (n - 1) / (n - 1 - k))
        except (ValueError, ZeroDivisionError): pass
        try:
            evaluations["RMSE"] = mth.sqrt(evaluations["MSE"])
        except ValueError: pass
        
        return evaluations
    
    def save_model(self, path: str, model_name: str = None, description: str = None,overwrite: bool = False)->None:
        """
        Saves the model architecture, parameters, optimizer state, and wrapper configuration to a file.

        The file is saved as a compressed symnn archive (``.symnn``).

        Parameters
        ----------
        path : str
            Directory path to save the file.
        model_name : str
            Name of the model that is going to be saved (will be used for the filename).
        description : str, optional
            Optional description to store in metadata.
        """
        if (model_name is None):
            model_name = self.model_name
        
        is_directory = os.path.isdir(path) or path.endswith("/") or path.endswith(os.sep)

        if is_directory:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{model_name}_{timestamp}.symnn"
            
            full_path = os.path.join(path, final_filename)
            
            # Note: Overwrite check omitted here because timestamps prevent collisions

        else:
            if not path.endswith(".symnn"):
                full_path = path + ".symnn"
            else:
                full_path = path

            # 3. The Overwrite Guardrail
            if (os.path.exists(full_path)):
                if not(overwrite):
                    temp = full_path[:-6]
                    offset = 1
                    while (os.path.exists(temp + f"_{offset}.symnn")):
                        offset += 1
                    full_path = temp + f"_{offset}.symnn"
        
        try:
            self.model.change_device("CPU")
            architecture_config = self.model.get_config()
            architecture_config.pop("network_structure")

            metadata = {
                'model_name': model_name,
                'model_class': str(self.model.__class__.__name__),
                'work_type': self.work_type,
                'description': description,
                'save_timestamp': datetime.datetime.now().isoformat(),
                'total_training_iterations': self.model.num_complited_train_iterations,
                'total_epochs_iterations':self.model.num_completed_epochs,
                'input_transformer':  self._input_transformer.__class__.__name__ if self._input_transformer else None,
                'output_transformer': self._output_transformer.__class__.__name__ if self._output_transformer else None,
                "loaded_train_data":self._loaded_train_data
            }

            transformation_configs = {}
            if self.input_transformer is not None:
                transformation_configs["inputs"] = self.input_transformer.get_config()
            if self.output_transformer is not None:
                transformation_configs["outputs"] = self.output_transformer.get_config()
            
            config_to_save = {
                'architecture': architecture_config,
                'metadata': metadata,
                'transformation_configs': transformation_configs,
                'optimizer_config':self.model._UPDATE_METHOD.get_config(),
            }
            

            params = self.model.get_parameters()

            flat_params = {}
            for layer_key, layer_params in params.items():
                for param_key, param_value in layer_params.items():
                    flat_params[f"{layer_key}_{param_key}"] = param_value
            
            opt_global, opt_layers = self.model._UPDATE_METHOD.get_state()


            for param_key, val in opt_global.items():
                flat_params[f"global_opt_{param_key}"] = val 

            for i, layer_opt_dict in enumerate(opt_layers):
                for param_key, matrix in layer_opt_dict.items():
                    flat_params[f"layer_{i}_opt_{param_key}"] = matrix

            npz_ram_buffer = io.BytesIO()
            np.savez_compressed(npz_ram_buffer, **flat_params)


            with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                
                archive.writestr("config.json", json.dumps(config_to_save, indent=4))
                archive.writestr("weights.npz", npz_ram_buffer.getvalue())

            return full_path
        except Exception as e:
            raise SavingError(f"Error al guardar el modelo.") from e

    def load_state(self, path: str) -> None:
        """
        Loads a model from a ``.symnn`` ZIP archive created by :meth:`save_model`.

        Reconstructs the Network, restores weights, optimizer state, and 
        data normalization statistics directly from RAM buffers.

        Parameters
        ----------
        path : str
            Path to the ``.symnn`` file.
        
        Raises
        ------
        IOError
            If the file cannot be loaded or has an invalid format.
        """
        model,input_transformer,output_transformer,metadata = self._extract_symnn_archive(path)
        self._model = model
        self._input_transformer = input_transformer,
        self._output_transformer = output_transformer
        self.model_name = metadata.get('model_name')
        self.work_type = metadata.get('work_type')
        
    @classmethod
    def load(cls, path: str) -> "Wrapper":
        """
        Creates a brand new Wrapper and populates it directly from a .symnn archive.
        
        Parameters
        ----------
        path : str
            Path to the .symnn archive.
        """
        model,input_transformer,output_transformer,metadata = cls._extract_symnn_archive(path)

        saved_work_type = metadata.get('work_type')
        # 2. Instantiate the blank Wrapper using the saved settings
        instance = cls(model=model, work_type=saved_work_type,input_transformer=input_transformer,output_transformer=output_transformer)

        # 3. Use Mode 2 to inject the architecture and math
        instance.model_name = metadata.get('model_name')
        return instance
    
    @staticmethod
    def _extract_symnn_archive(path: str) -> tuple[BaseNetwork,utilities.DataTransformer,utilities.DataTransformer,dict[str, any]]:
        """
        Internal method to load a .symnn archive.
        
        Parameters
        ----------
        path : str
            Path to the .symnn archive.
        """
        if not path.endswith(".symnn"):
            path = path + ".symnn"

        if not os.path.exists(path):
            raise PathError(f"No file found: {path}")

        try:
            with zipfile.ZipFile(path, 'r') as archive:
                # ==========================================
                # PHASE 1: THE SKELETON (JSON Metadata)
                # ==========================================
                config_bytes = archive.read("config.json")
                config_wrapper = json.loads(config_bytes)
                
                metadata = config_wrapper.get('metadata', {})
                transformation_configs = config_wrapper.get('transformation_configs', {})

                input_scaler_config = transformation_configs.get("inputs")
                if input_scaler_config is not None:
                    input_transformer = registries.registry.data_transformers_map[metadata["input_transformer"]]()
                    input_transformer.set_config(input_scaler_config)
                else:
                    input_transformer = None

                output_scaler_config = transformation_configs.get("outputs")
                if output_scaler_config is not None:
                    output_transformer = registries.registry.data_transformers_map[metadata["output_transformer"]]()
                    output_transformer.set_config(output_scaler_config)
                else:
                    output_transformer = None
                
                # 2. Reconstruct the Empty Architecture
                model_class_name = metadata.get('model_class')
                TargetNetworkClass = registries.registry._net_map[model_class_name]
                
                model = TargetNetworkClass.from_config(config_wrapper, registry_module=registries.registry)

                # ==========================================
                # PHASE 2: THE MATH (NPZ Binary Payload)
                # ==========================================
                npz_bytes = archive.read("weights.npz")
                npz_ram_buffer = io.BytesIO(npz_bytes)
                
                # The "Empty Buckets" for routing
                layer_params_dict = {}
                opt_global = {}
                opt_layers_dicts = [None]*len(model.layers)
                
                # We use allow_pickle=False because our matrices and scalars are pure!
                with np.load(npz_ram_buffer, allow_pickle=False) as data:
                    
                    for key, value in data.items():
                        
                        # Route A: Global Optimizer State (e.g., "global_opt_t")
                        if key.startswith("global_opt_"):
                            param_name = key.replace("global_opt_", "", 1)
                            # Convert 0-D numpy arrays back to pure Python scalars
                            opt_global[param_name] = value.item() if value.ndim == 0 else value
                            
                        # Route B: Layer Optimizer State (e.g., "layer_0_opt_m")
                        #NEED fix
                        elif "_opt_" in key:
                            parts = key.split("_opt_", 1)
                            layer_idx = int(parts[0].split("_")[1]) # Extracts integer 0
                            param_name = parts[1]                   # Extracts "m" or "v"
                            
                            if (opt_layers_dicts[layer_idx] is None):
                                opt_layers_dicts[layer_idx] = {}
                            opt_layers_dicts[layer_idx][param_name] = value
                            
                        # Route C: Layer Mathematical Parameters (e.g., "layer_0__weights")
                        elif key.startswith("layer_"):
                            # Safely split by the first two underscores: "layer" + "0" + "_weights"
                            parts = key.split("_", 2)
                            layer_key = f"{parts[0]}_{parts[1]}"  # Rebuilds "layer_0"
                            param_key = parts[2]                  # Rebuilds "_weights"
                            
                            if layer_key not in layer_params_dict:
                                layer_params_dict[layer_key] = {}
                            layer_params_dict[layer_key][param_key] = value

                # ==========================================
                # PHASE 3: CROSSING THE TRANSLATION BOUNDARY
                # ==========================================
                
                # 1. Inject pure dictionaries back into the Layers
                model.set_parameters(layer_params_dict)
                
                # 2. Inject pure Tuple/State back into the Optimizer
                # Note: We reconstruct the original dict format your optimizer expects
                rebuilt_opt_state = {"layer_states":opt_layers_dicts}
                rebuilt_opt_state.update(opt_global) # Adds 't'
                
                # Push state to Optimizer hardware
                model._UPDATE_METHOD.set_state(rebuilt_opt_state, model._CALCULATION_MANAGER)
                model._UPDATE_METHOD._initialize_state(model.layers)

                # 3. Restore Metadata Statistics
                model.num_complited_train_iterations = metadata.get('total_training_iterations', 0)
                model.num_completed_epochs = metadata.get('total_epochs_iteratios', 0)

                return (model, input_transformer, output_transformer, metadata)
        except Exception as e:
            raise LoadingError(f"Failed to load the model from {path}. The .symnn archive may be corrupted. Cause: {e}") from e


class GridSearchWrapper(Wrapper):
    """
    A wrapper that extends :class:`Wraper` to perform Hyperparameter Grid Search.

    Parameters
    ----------
    Model_class : Callable
        The class constructor to use for creating models (e.g., :class:`~HeteroSymNN.Core.Nets.neural_nets.SimpleNN`).
    work_type : Literal["class", "reg"]
        Type of problem (classification or regression).
    validation_testing_split : float, optional
        Fraction of data to use for validation during grid search (default 0.2).
    normalize_inputs : bool, optional
        Whether to normalize inputs.
    normalize_outputs : bool, optional
        Whether to normalize outputs.
    """
    def __init__(self,Model_class:type[BaseNetwork],work_type:Literal["class","reg"],validation_testing_split = 0.2,
                 normalize_inputs: bool = True, normalize_outputs: bool = True):
        
        super().__init__(model=None, work_type=work_type,
                         normalize_inputs=normalize_inputs, 
                         normalize_outputs=normalize_outputs)

        self.model_class = Model_class
        if not (0.0 < validation_testing_split < 1.0):
            raise ValueError("validation_split debe estar entre 0.0 y 1.0")
        self.VALIDATION_SPLIT = validation_testing_split

        self._X_train = None
        self._y_train = None
        self._X_vali = None
        self._y_vali = None
        self.best_model: BaseNetwork = None
        self.best_params: dict[str, any] = None
        self.best_score: float = -np.inf
        self.grid_search_results: list[dict[str, any]] = []

    def load_training(self, training_data: list, expected_results: list,shuffle:bool = True)->None:
        """
        Loads data and splits it into Training and Validation sets.

        Parameters
        ----------
        training_data : list
            All available input data.
        expected_results : list
            All available target data.
        shuffle : bool, optional
            Whether to shuffle data before splitting (default True).
        """
        # 1. Convertir a numpy para barajar y partir fácilmente
        X_full = np.array(training_data)
        Y_full = np.array(expected_results)

        if len(X_full) != len(Y_full):
            raise ValueError("Los datos de entrenamiento (X) y los resultados (Y) tienen diferente número de muestras.")
        
        # 2. Barajar los datos (en conjunto)
        indices = np.arange(X_full.shape[0])
        if (shuffle):
            np.random.shuffle(indices)
        
        X_shuffled = X_full[indices]
        Y_shuffled = Y_full[indices]
        
        # 3. Partir los datos
        split_idx = int(X_full.shape[0] * (1 - self.VALIDATION_SPLIT))
        
        if split_idx == 0 or split_idx == len(X_full):
            raise ValueError(f"El split de validación ({self.VALIDATION_SPLIT}) resulta en un conjunto de entrenamiento o validación vacío.")

        self._X_train = X_shuffled[:split_idx]
        self._y_train = Y_shuffled[:split_idx]
        self._X_vali = X_shuffled[split_idx:]
        self._y_vali = Y_shuffled[split_idx:]
        
 
        
        # Esto establece self.x_min, self.x_max, etc. usando SOLO el set de entrenamiento
        super().load_training(self._X_train, self._y_train)

    def _generate_param_combinations(self, param_grid: dict[str, list[any]]) -> list[dict[str, any]]:
        """
        Internal helper to generate all combinations of hyperparameters.
        
        Parameters
        ----------
        param_grid : dict[str, list[any]]
            Dictionary of the name of the parameter and the posible values.

        Returns
        -------
        list[dict[str, any]]
            List of dictionarys comprising all combinations of hyperparameters.
        """
        if not param_grid:
            return []

        param_keys = param_grid.keys()
        value_lists = param_grid.values()


        combinations_list = list(iter.product(*value_lists))
        combinations_dict = [dict(zip(param_keys, combo)) for combo in combinations_list]

        return combinations_dict

    def _run_grid_search(self,
                    static_params: dict[str, any],
                    param_grid: dict[str, list[any]],
                    metric_to_optimize: str = None,
                    higher_is_better: bool = True)->tuple[BaseNetwork, dict[str, any], list[dict[str, any]]]:
        """
        Internal method to execute the grid search loop.
        
        
        Parameters
        ----------
        static_params : dict[str, any]
            Dictionary of the parameters that will remain static during the grid search.
        param_grid : dict[str, list[any]]
            Dictionary of the name of the parameter and the posible values.
        metric_to_optimize : str, optional
            The metric name to use for selecting the best model.
        higher_is_better : bool, optional
            True if maximizing the metric, False if minimizing.
        """
        
        if (metric_to_optimize == None):
            if (self.work_type == "reg"):
                metric_to_optimize = "R2"
            else:
                metric_to_optimize = "Acur"
        
        combinations = self._generate_param_combinations(param_grid)
        
        self.best_score = -np.inf if higher_is_better else np.inf
        self.best_model = None
        self.best_params = None
        self.grid_search_results = []

        num_features = 0
        if (self._X_train is not None):
            num_features = self._X_train.shape[1]


        for i, params_combination in enumerate(combinations):
            readable_params = {}
            for k, v in params_combination.items():
                if isinstance(v, (optimizers.Optimizer,losses.Loss)):
                    readable_params[k] = v.__class__.__name__ + f"({v.get_config()})"
                elif inspect.isclass(v):
                    readable_params[k] = v.__name__
                else:
                    readable_params[k] = v

            start_time = time.time()
            
            try:
                model_args = static_params.copy()
                model_args.update(params_combination)
                
                self.model = self.model_class(**model_args)
                
                super().run_training()
                metrics:dict[str,float] = {}
                if (self.work_type == "reg"):
                    metrics = self.regreccion_test_accuracy(self._X_vali, self._y_vali)
                else:
                    metrics = self.classification_test_accuracy(self._X_vali,self._y_vali)
                score = metrics.get(metric_to_optimize)

                if (score == np.nan):
                    raise ValueError(f"Métrica '{metric_to_optimize}' no encontrada. Métricas disponibles: {metrics.keys()}")

                duration = time.time() - start_time
                
                result_entry = {'params': params_combination, 'score': score, 'metrics': metrics, 'duration_s': duration}
                self.grid_search_results.append(result_entry)

                if ((higher_is_better) and (score > self.best_score)):
                    self.best_score = score
                    self.best_model = self.model
                    self.best_params = params_combination

                elif ((not(higher_is_better)) and (score < self.best_score)):
                    self.best_score = score
                    self.best_model = self.model
                    self.best_params = params_combination


            except Exception as e:
                self.grid_search_results.append({'params': readable_params, 'score': None, 'error': str(e)})

        self.model = self.best_model
        return self.best_model, self.best_params, self.grid_search_results

    def run_training(self,
                     static_params: dict[str, any] = None,
                     param_grid: dict[str, list[any]] = None,
                     metric_to_optimize: str = 'R2',
                     higher_is_better: bool = True,
                     
                     num_iterations: int = None, 
                     training_mode: Literal["batch", "mini-batch", "stochastic"] = None, 
                     batch_size: int = None)->Union[list[float], tuple[BaseNetwork, dict[str, any], list[dict[str, any]]]]:
        """
        Executes training. Can function in two modes:

        1.  **Grid Search Mode:** If ``param_grid`` is provided. Iterates through all parameter combinations, 
            trains a new model for each, evaluates on the validation set, and selects the best one.
        2.  **Standard Training Mode:** If ``param_grid`` is None. Trains the currently active model (e.g., the best model found) 
            for additional iterations.

        Parameters
        ----------
        static_params : dict[str, any], optional
            Parameters that remain constant across all grid search trials (e.g., input size).
        param_grid : dict[str, list[any]], optional
            Dictionary where keys are parameter names and values are lists of possibilities to try.
        metric_to_optimize : str, optional
            The metric name to use for selecting the best model (e.g., 'R2', 'Acur', 'MSE').
        higher_is_better : bool, optional
            True if maximizing the metric (ejem. for Accuracy or R2), False if minimizing (ejem. for MSE).
        
        num_iterations : int, optional
            Number of training iterations. If None, uses the model's configured default.
        training_mode : Literal["batch", "mini-batch", "stochastic"], optional
            Training strategy. If None, uses the model's configured default.
        batch_size : int, optional
            Size of the batch when using "mini-batch" mode. If not pass, uses the model's configured default.
        
        Returns
        -------
        tuple or list
            In Grid Search mode: Returns ``(best_model, best_params, all_results)``.
            In Standard mode: Returns the loss history list.
        """
        
        if param_grid is not None:
            print("Grid search detectado (param_grid no es None). Iniciando búsqueda...")
            if ((self._X_train is None) or (self._y_train is None) or (self._X_vali is None) or (self._y_vali is None) or (self.model_class is None) or (static_params is None)):
                raise ValueError("Para grid search, debe proveer X_train, y_train, X_test, y_test, model_class, y static_params.")
            
            return self._run_grid_search(
                static_params=static_params,
                param_grid=param_grid,
                metric_to_optimize=metric_to_optimize,
                higher_is_better=higher_is_better
            )
            
        elif self.model is not None and num_iterations is not None:
            return super().run_training(
                num_iterations=num_iterations,
                training_mode=training_mode,
                batch_size=batch_size
            )
        
        elif self.model is None and num_iterations is not None:
            raise ValueError("No se ha encontrado un modelo. Debe ejecutar run_training con un 'param_grid' primero para encontrar un modelo.")
        
        else:
            raise ValueError("Argumentos insuficientes. Debe proveer un 'param_grid' (para grid search) o 'num_iterations' (para un modelo existente).")