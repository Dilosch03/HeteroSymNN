import os
import io
import numpy as np
import math as mth
import itertools as iter
from typing import Literal,Union,Optional
import datetime
import warnings
import copy
import time
import json
import zipfile
from pathlib import Path

from .. import __version__

from ..Core.Nets.base_classes import BaseNetwork
from ..Core import losses, optimizers
from . import data_transformers, registries
from ..exceptions import PathError,ShapeMismatchError,ShapeWarning,LoadingError,TrainingError,WrapperError,SavingError
from ..error_handlers import apply_clean_tracebacks
from ..utility import _NumpyEncoder

__all__ = ["Wrapper", "GridSearchManager"]

@apply_clean_tracebacks
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
    model : :class:`~HeteroSymNN.Core.Nets.BaseNetwork` | subclass of :class:`~HeteroSymNN.Core.Nets.BaseNetwork`
        The neural network instance to wrap.
    work_type : Literal["class", "reg"]
        The type of problem the model solves:
        
        *   ``"class"``: Classification (calculates Accuracy, F1, etc.).
        *   ``"reg"``: Regression (calculates MSE, R2, etc.).

    input_transformer : 'obj:`HeteroSymNN.API.DataTransformer`, optional
        The scaling method that is going to be use to scale the input data. Defaults to ``None`` and doesn't scale the data.
    output_transformer : 'obj:`HeteroSymNN.API.DataTransformer`, optional
        The descaling method that is going to be use to scale the input data. Defaults to ``None`` and doesn't descale the data.

    Attributes
    ----------
    work_type: str read-write
        The type of problem the model solves.
    
    """
    def __init__(self, model: BaseNetwork,work_type:Literal["class","reg"],input_transformer: Optional[data_transformers.DataTransformer] = None, output_transformer: Optional[data_transformers.DataTransformer] = None, shuffle_samples: bool = False):
        if not(issubclass(type(model), BaseNetwork)):
            raise WrapperError("The model that was pass is not a subclass of BaseNetwork.")
        
        self._model:BaseNetwork = model
        self.training_data = None
        self._input_transformer = input_transformer
        self._output_transformer = output_transformer
        self.work_type = work_type
        self._loaded_train_data = False
        self.model_name = model.__class__.__name__
        self.shuffle_samples = shuffle_samples

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
                warnings.warn(f"The model new is expecting {expected_x} features, but the loaded data has {X_norm.shape[1]} features.",ShapeMismatchError,stacklevel=2)
            expected_y = new_model.layers[-1].num_nodes
            if Y_norm.ndim == 2 and Y_norm.shape[1] != expected_y:
                warnings.warn(f"The model new is expecting {expected_y} features, but the loaded data has {Y_norm.shape[1]} targets.",ShapeMismatchError,stacklevel=2)
            
    @property
    def input_transformer(self)->Union[data_transformers.DataTransformer, None]:
        """
        The transformer used to scale input data.
        """
        return self._input_transformer

    @input_transformer.setter
    def input_transformer(self, new_transformer: data_transformers.DataTransformer):
        self._input_transformer = new_transformer
        if (self._loaded_train_data):
            self.training_data_norm[0] = new_transformer.fit_transform(self.training_data[0])

    @property
    def output_transformer(self)->Union[data_transformers.DataTransformer, None]:
        """
        The transformer used to scale output data.
        """
        return self._output_transformer
    
    @output_transformer.setter
    def output_transformer(self, new_transformer: data_transformers.DataTransformer):
        self._output_transformer = new_transformer
        if(self._loaded_train_data):
            self.training_data_norm[1] = new_transformer.fit_transform(self.training_data[1])

    def fit(self, training_data: list, expected_results: list, epochs: int = None, batch_size: int = None, return_batch_losses: bool = False)->list[float]:
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
        batch_size : int, optional
            Size of the batch. If not pass, uses the model's configured default.
        return_batch_losses: bool, optional
            Whether to return the loss for each batch or the average loss for each epoch.
            Default is False.

        Returns
        -------
        list[float]
            A list of loss values recorded during training.
            If ``return_batch_losses`` is True, it returns the loss for each batch.
            If ``return_batch_losses`` is False, it returns the average loss for each epoch.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ShapeMismatchError`
            If dimensions do not match the model's expected input/output size.
        """
        self.load_training(training_data, expected_results)
        return self.run_training(epochs, batch_size=batch_size, return_batch_losses=return_batch_losses)

    def load_training(self, training_data: list, expected_results: list)->None:
        """
        Loads and prepares training data by doing the following steps:
        
        1.  Converts inputs to NumPy arrays.
        2.  Checks dimensions against the model architecture.
        3.  Calculates transformation of the given data if a transformer was pass.
        4.  Stores the normalized data for training.

        Parameters
        ----------
        training_data : list or np.ndarray
            Input features (X). Shape should be (n_samples, n_features).
        expected_results : list or np.ndarray
            Target labels/values (Y). Shape should be (n_samples, n_outputs).
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.ShapeMismatchError`
            If dimensions do not match the model's expected input/output size.
        """
        self._loaded_train_data = True
        X_raw = np.array(training_data)
        Y_raw = np.array(expected_results)
        
        if X_raw.ndim == 1:
            warnings.warn("The shape of the inputs are 1D. Resheaping it to (num samples, 1).",ShapeWarning,stacklevel=2)
            X_raw = X_raw.reshape(-1, 1)
            
        if Y_raw.ndim == 1:
            warnings.warn("The shape of the outputs are 1D. Resheaping it to (num samples, 1).",ShapeWarning,stacklevel=2)
            Y_raw = Y_raw.reshape(-1, 1)
        
        if (self._model != None):
            expected_x_features = self.model.layers[0].num_inputs
            if X_raw.ndim == 2 and X_raw.shape[1] != expected_x_features:
                raise ShapeMismatchError(
                    f"Input matrix orientation mismatch. Network expects {expected_x_features} features, "
                    f"but got {X_raw.shape[1]}."
                )

            expected_y_features = self.model.layers[-1].num_nodes      
            if Y_raw.ndim == 2 and Y_raw.shape[1] != expected_y_features:
                raise ShapeMismatchError(f"Target shape mismatch. The network's final layer is configured to output {expected_y_features} values, but the target data (y) provides {Y_raw.shape[1]} values. Ensure y is shaped (n_samples, {expected_y_features})")
            
        if (X_raw.shape[0] != Y_raw.shape[0]):
            raise ShapeMismatchError(f"Sample count mismatch. The training features (X) contain {X_raw.shape[0]} samples, but the training targets (y) contain {Y_raw.shape[0]} samples. Both matrices must have an identical number of rows.")
        
        self.training_data = (X_raw, Y_raw)
        X_norm = X_raw

        if (self._input_transformer is not None):
            X_norm = self._input_transformer.fit_transform(X_raw)

        Y_norm = Y_raw
        if (self._output_transformer is not None):
            Y_norm = self._output_transformer.fit_transform(Y_raw)
            
        
        self.training_data_norm = (X_norm, Y_norm)

    def run_training(self, num_iterations:int = None, batch_size: int = None, return_batch_losses: bool = False)->list[float]:
        """
        Executes the training loop on the loaded data.

        Parameters
        ----------
        num_iterations : int, optional
            Number of epochs to train. If None, uses the model's configured default.
        batch_size : int, optional
            Size of the batch. If not pass, uses the model's configured default.
        return_batch_losses: bool, optional
            Whether to return the loss for each batch or the average loss for each epoch.
            Default is False.

        Returns
        -------
        list[float]
            A list of loss values recorded during training.
            If ``return_batch_losses`` is True, it returns the loss for each batch.
            If ``return_batch_losses`` is False, it returns the average loss for each epoch.
        """
        if self.training_data is None:
            raise TrainingError("No Training data loaded. Use load_training() first.")
            
        self.model.to("device")
        
        # Keep full dataset as NumPy on CPU, slice there, cast batches later
        train_data = self.training_data_norm[0].T
        train_targets = self.training_data_norm[1].T
        
        num_samples = self.training_data_norm[0].shape[0]
        
        b_size = self.model.batch_size if batch_size is None else batch_size
        if num_iterations is None:
            num_iterations = self.model.num_training_epochs

        if b_size == -1:
            b_size = num_samples

        if b_size != self.model.batch_size:
            self.model._BATCH_SIZE = b_size
            for layer in self.model.layers:
                layer.batch_size_change(b_size)
        
        batch_losses_list = []
        for _ in range(num_iterations):
            self.model.num_completed_epochs += 1
            iter_loss = self.model._CALCULATION_MANAGER.array(0.0, dtype=self.model._DEFAULT_FLOAT_TYPE)
            
            # Using numpy since train_data is on CPU
            if self.shuffle_samples:
                indices = np.random.permutation(num_samples)
            else:
                indices = np.arange(num_samples)

            for start_idx in range(0, num_samples, b_size):
                end_idx = min(start_idx + b_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                x_batch = train_data[:,batch_indices]
                y_batch = train_targets[:,batch_indices]
                
                # 1. Hardware abstraction: Send data to current backend (CPU/GPU)
                x, y = self.model.cast_arrays(x_batch, y_batch)
                
                # 2. Forward pass
                y_pred = self.model.forward(x)
                
                # 3. Loss Calculation
                loss_val = self.model.loss_function.forward(y_pred, y)
                error = self.model.loss_function.backward(y_pred, y)
                
                # 4. Backward Pass
                self.model.backward(error)
                
                # 5. Parameter Update
                self.model.update_params()
                self.model.num_completed_train_iterations += 1
                
                if return_batch_losses:
                    batch_losses_list.append(float(self.model._ASNUMPY(loss_val)))

                iter_loss += loss_val * (end_idx - start_idx)

            avg_loss = float(self.model._ASNUMPY(iter_loss) / num_samples)
            self.model.history_losses.append(avg_loss)

        if return_batch_losses:
            return batch_losses_list
        return self.model.history_losses
    
    def predict(self, data: list)->np.ndarray:
        """
        Generates predictions for new data.

        Handles input and output transformations automatically if configured.

        Parameters
        ----------
        data : list or np.ndarray
            Input features to predict.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If there is a state problem with the DataTransformer objects.
        
        """
        X_raw = np.array(data)
        
        X_norm = X_raw
        if (self._input_transformer is not None):
            None_keys = []
            data_trans_config = self._input_transformer.get_config()
            for key in data_trans_config.keys():
                if data_trans_config[key] is None:
                    None_keys.append(key)

            if not(self._input_transformer.is_fitted):
                raise WrapperError("A DataTransform object was pass but it was never fitted.")
            
            if (len(None_keys) > 0):
                raise WrapperError(f"Input transformer has {','.join(None_keys)} with None values even though it was fitted.")
                
            X_norm = self._input_transformer.transform(X_raw)
            
            
        Y_pred_norm = self.model.predict(X_norm)
        
        Y_denorm = Y_pred_norm
        if (self._output_transformer is not None):
            None_keys = []
            data_trans_config = self._output_transformer.get_config()
            for key in data_trans_config.keys():
                if data_trans_config[key] is None:
                    None_keys.append(key)

            if not(self._output_transformer.is_fitted):
                raise WrapperError("A DataTransform object was pass but it was never fitted.")
            
            if (len(None_keys) > 0):
                raise WrapperError(f"Output transformer has {','.join(None_keys)} with None values even though it was fitted.")
                
            Y_denorm = self._output_transformer.inverse_transform(Y_pred_norm)
        
        return Y_denorm

    def test_accuracy(self,test_data:list,expected_results:list,threshold:float = 0.5)->Union[dict[str, float], tuple[dict[str, float], dict[str, int]]]:
        """
        Evaluates the model on the given test dataset.

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
            
        *   `reg`: dict[str,float]
        *   `class`: tuple[dict[str, float], dict[str, int]

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If the WorkType attribute is not set to 'reg' or 'class'.
        """
        results = {}
        if (self.work_type == "reg"):
            results = self.regression_test_accuracy(test_data,expected_results)
        elif(self.work_type == "class"):
            results = self.classification_test_accuracy(test_data,expected_results,threshold)
        else:
            raise WrapperError("Work type was not specified.")
        
        return results

    def classification_test_accuracy(self, test_data: list, expected_results: list,threshold:float = 0.5)->tuple[dict[str, float], dict[str, int]]:
        """
        Calculates classification metrics (Accuracy, Precision, Recall/TPR, F1 Score).

        Note: Currently assumes binary classification or multi-label, default threshold is 0.5.

        Parameters
        ----------
        test_data : list or np.ndarray
            Input features for testing.
        expected_results : list or np.ndarray
            Ground truth values.
        threshold : float, optional
            Threshold for binary classification. Default is 0.5.

        Returns
        -------
        tuple[dict[str, float], dict[str, int]]
            A tuple containing:
            
            1.  Dictionary of metrics (Acur, Press, TPR, F1).
            2.  Dictionary of raw counts (TP, TN, FP, FN).
        """
        predictions_raw = self.predict(test_data)
        expected_results = np.array(expected_results)
        
        if predictions_raw.ndim == 1:
            predictions_raw = predictions_raw.reshape(-1, 1)
        if expected_results.ndim == 1:
            expected_results = expected_results.reshape(-1, 1)
            
        predictions = (predictions_raw > threshold).astype(int)
        
        results_compare = {"correc_pos":0, "correct_neg":0, "false_pos":0, "false_neg":0}

        for i in range(expected_results.shape[0]):
            for j in range(expected_results.shape[1]):
                is_equal = (predictions[i, j] == expected_results[i, j])
                if(is_equal):
                    if(predictions[i, j]):
                        results_compare["correc_pos"] += 1
                    else:
                        results_compare["correct_neg"] += 1
                else:
                    if(predictions[i, j]):
                        results_compare["false_pos"] += 1
                    else:
                        results_compare["false_neg"] += 1
        
        evals = {"Acur":-1, "Press":-1, "TPR":-1, "F1":-1}
        correct = results_compare["correc_pos"] + results_compare["correct_neg"]
        total = expected_results.size
        
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
            if (evals["Press"] >= 0 and evals["TPR"] >= 0):
                evals["F1"] = 2 * evals["Press"] * evals["TPR"] / (evals["Press"] + evals["TPR"])
        except ZeroDivisionError: pass

        return (evals, results_compare)
    
    def regression_test_accuracy(self, test_data: list, expected_results: list)->dict[str, float]:
        """
        Calculates regression metrics (R2, MSE, RMSE, MAE, MAPE, AIC, BIC).

        Parameters
        ----------
        test_data : list or np.ndarray
            Input features for testing.
        expected_results : list or np.ndarray
            Ground truth values.

        Returns
        -------
        dict[str, float]
            Dictionary containing the calculated metrics.
        """
        predictions_raw = self.predict(test_data)
        expected_results = np.array(expected_results)
        
        if predictions_raw.ndim == 1:
            predictions_raw = predictions_raw.reshape(-1, 1)
        if expected_results.ndim == 1:
            expected_results = expected_results.reshape(-1, 1)
            
        results = predictions_raw
        
        if (results.shape != expected_results.shape):
            raise ShapeMismatchError(f"The expected results are not in the expected shape, Received {expected_results.shape} and expected {results.shape}.")

        mean = np.mean(expected_results)
        rss = np.sum((expected_results - results)**2)
        ssr = np.sum((results - mean)**2)
        mae = np.sum(np.abs(expected_results - results))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_array = np.where(expected_results != 0, np.abs((expected_results - results) / expected_results), 0)
            mape = np.sum(mape_array)

        if (np.isnan(mape)):
            mape = 0
        
        tss = rss + ssr
        n = expected_results.size
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
    
    def save_model(self, path: str, model_name: str = None, description: str = None,overwrite: bool = False)->str:
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
        overwrite : bool, optional
            Whether to overwrite the file if it already exists.

        Returns
        -------
        str
            The path to the saved file.

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If the file cannot be saved.
        
        """
        if (model_name is None):
            model_name = self.model_name
        
        if  (path.startswith("/")):
            path = os.path.join(os.getcwd(), path)
        
        is_directory = os.path.isdir(path) or path.endswith("/") or path.endswith(os.sep)

        if is_directory:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{model_name}_{timestamp}.symnn"
            
            full_path = os.path.join(path, final_filename)
            

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

            full_path = Path(full_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.model.to("host")
            config_to_save,flat_params= self.model._get_save_objects(model_name, description)

            metadata_extra = {
                'work_type': self.work_type,
                'input_transformer':  self._input_transformer.__class__.__name__ if self._input_transformer else None,
                'output_transformer': self._output_transformer.__class__.__name__ if self._output_transformer else None,
                "loaded_train_data":self._loaded_train_data,
                "shuffle_samples": self.shuffle_samples
            }


            transformation_configs = {}
            if self.input_transformer is not None:
                transformation_configs["inputs"] = self.input_transformer.get_config()
            if self.output_transformer is not None:
                transformation_configs["outputs"] = self.output_transformer.get_config()
            
            config_to_save.update({
                'transformation_configs': transformation_configs,
                'optimizer_config':self.model._UPDATE_METHOD.get_config(),
            })
            config_to_save["metadata"].update(metadata_extra)


            npz_ram_buffer = io.BytesIO()
            np.savez_compressed(npz_ram_buffer, **flat_params)

            with zipfile.ZipFile(full_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                
                archive.writestr("config.json", json.dumps(config_to_save, indent=4, cls=_NumpyEncoder))
                archive.writestr("weights.npz", npz_ram_buffer.getvalue())

            return full_path
        except Exception as e:
            raise SavingError(f"Error ocured when saving the model {self.model_name}. {str(e)}")

    def load_state(self, path: str) -> None:
        """
        Loads a model from a ``.symnn`` ZIP archive created by :meth:`save_model` and ovewrittes the current paramteres of the model and wrapper.

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
        self._input_transformer = input_transformer
        self._output_transformer = output_transformer
        self.model_name = metadata.get('model_name')
        self.work_type = metadata.get('work_type')
        self.shuffle_samples = metadata.get('shuffle_samples', False)
        
    @classmethod
    def load_model(cls, path: str) -> "Wrapper":
        """
        Creates a brand new Wrapper and populates it directly from a .symnn archive.
        
        Parameters
        ----------
        path : str
            Path to the .symnn archive.
        """
        model,input_transformer,output_transformer,metadata = cls._extract_symnn_archive(path)

        saved_work_type = metadata.get('work_type')
        saved_shuffle_samples = metadata.get('shuffle_samples', False)
        instance = cls(model=model, work_type=saved_work_type,input_transformer=input_transformer,output_transformer=output_transformer, shuffle_samples=saved_shuffle_samples)

        instance.model_name = metadata.get('model_name')
        return instance
    
    @staticmethod
    def _extract_symnn_archive(path: str) -> tuple[BaseNetwork,data_transformers.DataTransformer,data_transformers.DataTransformer,dict[str, any]]:
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
                

                model_class_name = metadata.get('model_class')
                TargetNetworkClass = registries.registry._net_map[model_class_name]
                
                model = TargetNetworkClass.from_config(config_wrapper, registry_module=registries.registry)

                npz_bytes = archive.read("weights.npz")
                npz_ram_buffer = io.BytesIO(npz_bytes)
                
                layer_params_dict = {}
                opt_global = {}
                opt_layers_dicts = [None]*len(model.layers)
                
                with np.load(npz_ram_buffer, allow_pickle=False) as data:
                    for key, value in data.items():
                        if key.startswith("global_opt_"):
                            param_name = key.replace("global_opt_", "", 1)
                            # Convert 0-D numpy arrays back to pure Python scalars
                            opt_global[param_name] = value.item() if value.ndim == 0 else value

                        elif "_opt_" in key:
                            parts = key.split("_opt_", 1)
                            layer_idx = int(parts[0].split("_")[1])
                            param_name = parts[1]                   
                            
                            if (opt_layers_dicts[layer_idx] is None):
                                opt_layers_dicts[layer_idx] = {}
                            opt_layers_dicts[layer_idx][param_name] = value
                            
                        elif key.startswith("layer_"):
                            parts = key.split("_", 2)
                            layer_key = f"{parts[0]}_{parts[1]}"
                            param_key = parts[2]                 
                            
                            if layer_key not in layer_params_dict:
                                layer_params_dict[layer_key] = {}
                            layer_params_dict[layer_key][param_key] = value
                
                model.set_parameters(layer_params_dict)
                
                rebuilt_opt_state = {"layer_states":opt_layers_dicts}
                rebuilt_opt_state.update(opt_global)

                model._UPDATE_METHOD.set_state(rebuilt_opt_state, model._CALCULATION_MANAGER)
                model._UPDATE_METHOD._initialize_state(model.layers)

                model.num_completed_train_iterations = metadata.get('total_training_iterations', 0)
                model.num_completed_epochs = metadata.get('total_epochs_iterations', 0)

                return (model, input_transformer, output_transformer, metadata)
        except Exception as e:
            raise LoadingError(f"Failed to load the model from {path}. The .symnn archive may be corrupted. Cause: {e}") from e



class GridSearchManager:
    """
    An independent orchestrator that performs Hyperparameter Grid Search using Composition.
    
    The manager takes a template Wrapper, 
    mutates its architectural DNA based on a parameter grid, spawns completely independent 
    clones, and compares them to find the optimal configuration using a specified metric.

    Parameters
    ----------
    template_wrapper : :class:`~HeteroSymNN.API.wrappers.Wrapper`
        A fully initialized Wrapper that serves as the base blueprint.
    param_grid : dict[str, list[any]]
        Dictionary where keys are parameter names and values are lists of possibilities to try.
    validation_split : float, optional
        Fraction of data to use for validation during grid search (default 0.2).
    """
    def __init__(self, template_wrapper: Wrapper, param_grid: dict[str, list[any]], validation_split: float = 0.2):
        self.template_wrapper = template_wrapper
        self.param_grid = param_grid
        
        if not (0.0 < validation_split < 1.0):
            raise WrapperError("validation_split value should be between 0 and 1.")
        self.validation_split = validation_split
        
        self._X_train, self._y_train = None, None
        self._X_vali, self._y_vali = None, None
        
        self.best_wrapper: Wrapper = None
        self.best_params: dict[str, any] = None
        self.best_score: float = None
        self.grid_search_results: list[dict[str, any]] = []

    def load_data(self, training_data: list, expected_results: list, shuffle: bool = True) -> None:
        """
        Loads data and splits it into Training and Validation sets.

        Parameters
        ----------
        training_data : list
            The training data.
        expected_results : list
            The expected results.
        shuffle : bool, optional
            Whether to shuffle the data (default True).

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If the validation split is not between 0 and 1.
        :exc:`~HeteroSymNN.exceptions.ShapeMismatchError`
            If the number of samples in input and output data do not match.
        """
        X_full = np.array(training_data)
        Y_full = np.array(expected_results)

        if len(X_full) != len(Y_full):
            raise ShapeMismatchError(f"Number of samples in input and output data do not match. input samples: {len(X_full)}, output samples: {len(Y_full)}")
        
        indices = np.arange(X_full.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        
        X_shuffled = X_full[indices]
        Y_shuffled = Y_full[indices]
        
        split_idx = int(X_full.shape[0] * (1 - self.validation_split))
        
        if split_idx == 0 or split_idx == len(X_full):
            raise WrapperError(f"The value for the split for validation ({self.validation_split}) returns and empty split for one of the tasks.")

        self._X_train = X_shuffled[:split_idx]
        self._y_train = Y_shuffled[:split_idx]
        self._X_vali = X_shuffled[split_idx:]
        self._y_vali = Y_shuffled[split_idx:]

    def _generate_param_combinations(self) -> list[dict[str, any]]:
        """
        Internal helper to generate all combinations of hyperparameters.

        Returns
        -------
        list[dict[str, any]]
            List of parameter combinations.
        """
        if not self.param_grid:
            return []
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, combo)) for combo in iter.product(*values)]

    def _clone_wrapper(self, new_config: dict[str,any]) -> Wrapper:
        """
        Internal helper to dynamically mutate the cofiguration of the template and spawn a fresh wrapper clone.

        Parameters
        ----------
        new_config: dict[str, any]
            the new congfiguration that is going to be applied to the template.

        Returns
        -------
        :class:`~HeteroSymNN.API.wrappers.Wrapper`
            Fresh wrapper with the mutated configuration.
        
        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If the model cannot be cloned.
        """
        self.template_wrapper.model.to("host")
        architecture_config = self.template_wrapper.model.get_config()

        base_config = {
            'architecture': architecture_config,
            'metadata': {
                'model_class': self.template_wrapper.model.__class__.__name__,
                'work_type': self.template_wrapper.work_type,
                'input_transformer': self.template_wrapper.input_transformer.__class__.__name__ if self.template_wrapper.input_transformer else None,
                'output_transformer': self.template_wrapper.output_transformer.__class__.__name__ if self.template_wrapper.output_transformer else None,
                'shuffle_samples': self.template_wrapper.shuffle_samples,
            },
            'transformation_configs': {},
            'optimizer_config': self.template_wrapper.model._UPDATE_METHOD.get_config(),
        }
        if self.template_wrapper.input_transformer is not None:
            base_config['transformation_configs']['inputs'] = self.template_wrapper.input_transformer.get_config()
        if self.template_wrapper.output_transformer is not None:
            base_config['transformation_configs']['outputs'] = self.template_wrapper.output_transformer.get_config()

        mutated = copy.deepcopy(base_config)
        
        for k, v in new_config.items():
            if k in mutated['architecture']:
                mutated['architecture'][k] = v
            elif k in mutated['optimizer_config']:
                mutated['optimizer_config'][k] = v
            elif 'loss_config' in mutated['architecture'] and k in mutated['architecture']['loss_config']:
                mutated['architecture']['loss_config'][k] = v
            elif k == 'learning_rate': 
                mutated['optimizer_config']['learning_rate'] = v

        in_scaler, out_scaler = None, None
        meta = mutated['metadata']
        t_configs = mutated['transformation_configs']
        
        if meta.get('input_transformer'):
            in_scaler = registries.registry.data_transformers_map[meta['input_transformer']]()
            if "inputs" in t_configs: in_scaler.set_config(t_configs["inputs"])
                
        if meta.get('output_transformer'):
            out_scaler = registries.registry.data_transformers_map[meta['output_transformer']]()
            if "outputs" in t_configs: out_scaler.set_config(t_configs["outputs"])

        TargetClass = registries.registry.net_map[meta['model_class']]
        new_model = TargetClass.from_config(mutated, registry_module=registries.registry)

        new_wrapper = Wrapper(
            model=new_model,
            work_type=meta['work_type'],
            input_transformer=in_scaler,
            output_transformer=out_scaler,
            shuffle_samples=meta.get('shuffle_samples', False)
        )
        new_wrapper.load_training(self._X_train, self._y_train)
        return new_wrapper

    def execute_search(self, metric_to_optimize: str = None, higher_is_better: bool = True) -> tuple[Wrapper, dict, list]:
        """
        Executes the grid search loop across all parameter combinations.
        
        Parameters
        ----------
        metric_to_optimize : str, optional
            The metric name to use for selecting the best model (e.g., 'R2', 'Acur').
        higher_is_better : bool, optional
            True if maximizing the metric, False if minimizing.
        
        Returns
        -------
        tuple
            Returns ``(best_wrapper, best_params, all_results)``.

        Raises
        ------
        :exc:`~HeteroSymNN.exceptions.WrapperError`
            If the trainig data was not loaded before the search.
        
        """
        if self._X_train is None:
            raise WrapperError("No data loaded. Call load_data() before execute_search().")

        if metric_to_optimize is None:
            metric_to_optimize = "R2" if self.template_wrapper.work_type == "reg" else "Acur"

        self.best_score = -np.inf if higher_is_better else np.inf
        self.best_wrapper = None
        self.best_params = None
        self.grid_search_results = []

        combinations = self._generate_param_combinations()

        for combo in combinations:
            start_time = time.time()
            readable_combo = {k: (v.__class__.__name__ if isinstance(v, (optimizers.Optimizer, losses.Loss)) else v) for k, v in combo.items()}
            
            try:
                trial_wrapper = self._clone_wrapper(combo)
                trial_wrapper.run_training() 
                
                metrics = {}
                if trial_wrapper.work_type == "reg":
                    metrics = trial_wrapper.regression_test_accuracy(self._X_vali, self._y_vali)
                else:
                    metrics, _ = trial_wrapper.classification_test_accuracy(self._X_vali, self._y_vali)
                    
                score = metrics.get(metric_to_optimize)
                
                if score is None or np.isnan(score):
                    raise WrapperError(f"Metric '{metric_to_optimize}' not found or NaN. Available: {list(metrics.keys())}")
                    
                duration = time.time() - start_time
                self.grid_search_results.append({'params': readable_combo, 'score': score, 'metrics': metrics, 'duration_s': duration})

                if (higher_is_better and score > self.best_score) or (not higher_is_better and score < self.best_score):
                    self.best_score = score
                    self.best_wrapper = trial_wrapper
                    self.best_params = combo

            except Exception as e:
                self.grid_search_results.append({'params': readable_combo, 'score': None, 'error': str(e)})

        return self.best_wrapper, self.best_params, self.grid_search_results