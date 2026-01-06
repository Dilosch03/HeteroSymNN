import os
import numpy as np
import math as mth
import itertools as iter
from typing import Literal,Callable,Union
import datetime
import warnings
import inspect
import time


from ..Backend import hardware as HW
from ..Core.Nets.dense import HeteroDense
from ..Core import losses, optimizers,initializers
from . import registry

def _normalization(vals: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """Internal helper for Min-Max normalization."""
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (vals - min_val) / range_val

def _denormalization(vals: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """Internal helper for Min-Max denormalization."""
    return vals * (max_val - min_val) + min_val

class Wrapper():
    """
    A high-level wrapper for managing the lifecycle of a :class:`~HeteroSymNN.Core.Nets.neural_nets.HeteroDense`, :class:`~HeteroSymNN.Core.Nets.neural_nets.FlexibleNN` or :class:`~HeteroSymNN.Core.Nets.neural_nets.SimpleNN`

    This class simplifies common tasks such as:
    
    *   Data loading and normalization (Min-Max scaling).
    *   Training execution and loss tracking.
    *   Model evaluation (Classification metrics or Regression metrics).
    *   Saving and loading the full model state (architecture, weights, optimizer state).

    Parameters
    ----------
    model : :class:`~HeteroSymNN.Core.Nets.neural_nets.HeteroDense` | :class:`~HeteroSymNN.Core.Nets.neural_nets.FlexibleNN` | :class:`~HeteroSymNN.Core.Nets.neural_nets.SimpleNN` or None
        The neural network instance to wrap. Can be ``None`` if loading a model from disk later.
    work_type : Literal["class", "reg"]
        The type of problem the model solves:
        
        *   ``"class"``: Classification (calculates Accuracy, F1, etc.).
        *   ``"reg"``: Regression (calculates MSE, R2, etc.).
    normalize_inputs : bool, optional
        If ``True``, input data (X) will be automatically normalized to [0, 1] based on training data statistics. Defaults to ``True``.
    normalize_outputs : bool, optional
        If ``True``, output data (Y) will be automatically normalized to [0, 1] during training and denormalized during prediction. Defaults to ``True``.
    """
    def __init__(self, model: HeteroDense,work_type:Literal["class","reg"],normalize_inputs: bool = True, normalize_outputs: bool = True):
        self._model:HeteroDense = None
        self.training_data = None
        self.training_data_norm = None
        self.normalize_inputs = normalize_inputs 
        self.normalize_outputs = normalize_outputs
        self.work_type =work_type
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self._loaded_train_data = False

        self.model:HeteroDense = model

    @property
    def model(self)->HeteroDense:
        """
        The underlying neural network instance.
        
        When setting this property, the wrapper validates that the new model's input/output dimensions 
        match any previously loaded training data.
        """
        return self._model
    
    @model.setter
    def model(self, new_model):
        # 1. Asignar el modelo
        self._model = new_model
        
        # 2. Validación Automática: Si hay datos cargados y el modelo no es None, verificar compatibilidad
        if ((new_model != None) and (self._loaded_train_data)):
            X_norm, Y_norm = self.training_data_norm
            
            # Verificar dimensiones de entrada
            expected_x = new_model.layers[0].num_inputs
            # Asumimos que X_norm tiene forma (n_muestras, n_features) tras load_training
            if X_norm.ndim == 2 and X_norm.shape[1] != expected_x:
                raise ValueError(f"Error de Arquitectura: El nuevo modelo espera {expected_x} entradas, pero los datos cargados tienen {X_norm.shape[1]} columnas (features).")
            
            # Verificar dimensiones de salida
            expected_y = new_model.layers[-1].num_nodes
            if Y_norm.ndim == 2 and Y_norm.shape[1] != expected_y:
                raise ValueError(f"Error de Arquitectura: El nuevo modelo espera {expected_y} salidas, pero los datos cargados tienen {Y_norm.shape[1]} columnas (targets).")

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
            warnings.warn("Los datos de entrada (X) eran 1D. Remodelando a (n_muestras, 1).")
            X_raw = X_raw.reshape(-1, 1)
            
        if Y_raw.ndim == 1:
            warnings.warn("Los datos de salida (Y) eran 1D. Remodelando a (n_muestras, 1).")
            Y_raw = Y_raw.reshape(-1, 1)
        
        if (self._model != None):
            expected_x_features = self.model.layers[0].num_inputs
            if X_raw.ndim == 2 and X_raw.shape[1] != expected_x_features:
                if X_raw.shape[0] == expected_x_features:
                    warnings.warn(f"Los datos de entrada (X) parecían estar en formato (n_features, n_muestras). Transponiendo a (n_muestras, n_features).", UserWarning)
                    X_raw = X_raw.T
                else:
                    raise ValueError(f"La forma de los datos de entrada (X) es {X_raw.shape}, pero la capa 0 espera {expected_x_features} features (en la segunda dimensión).")

            expected_y_features = self.model.layers[-1].num_nodes      
            if Y_raw.ndim == 2 and Y_raw.shape[1] != expected_y_features:
                if Y_raw.shape[0] == expected_y_features:
                    warnings.warn(f"Los datos de salida (Y) parecían estar en formato (n_outputs, n_muestras). Transponiendo a (n_muestras, n_outputs).", UserWarning)
                    Y_raw = Y_raw.T 
                else:
                    raise ValueError(f"La forma de los datos de salida (Y) es {Y_raw.shape}, pero la capa final espera {expected_y_features} outputs (en la segunda dimensión).")
            
        self.training_data = (X_raw, Y_raw)
        
        if self.normalize_inputs:
            self.x_min = np.min(X_raw, axis=0)
            self.x_max = np.max(X_raw, axis=0)
            X_norm = _normalization(X_raw, self.x_min, self.x_max)
        else:
            self.x_min = None
            self.x_max = None
            X_norm = X_raw

        if self.normalize_outputs:
            self.y_min = np.min(Y_raw, axis=0)
            self.y_max = np.max(Y_raw, axis=0)
            Y_norm = _normalization(Y_raw, self.y_min, self.y_max)
        else:
            self.y_min = None 
            self.y_max = None
            Y_norm = Y_raw
        
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
            raise ValueError("No hay datos de entrenamiento cargados. Usa load_training().")
        
        losses = self.model.train(
            self.training_data_norm[0], # X_norm
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
        
        if self.normalize_inputs:
            if self.x_min is None:
                 raise ValueError("El modelo fue entrenado con normalize_inputs=True pero no se encontraron estadísticas (x_min). ¿Fue cargado correctamente?")
            X_norm = _normalization(X_raw, self.x_min, self.x_max)
        else:
            X_norm = X_raw
            
        Y_pred_norm = self.model.predict(X_norm)
        
        if self.normalize_outputs:
            if self.y_min is None:
                raise ValueError("El modelo fue entrenado con normalize_outputs=True pero no se encontraron estadísticas (y_min). ¿Fue cargado correctamente?")
            Y_denorm = _denormalization(Y_pred_norm, self.y_min, self.y_max)
            return Y_denorm
        else:
            return Y_pred_norm 

    def test_accuracy(self,test_data:list,expected_results:list)->Union[dict[str, float], tuple[dict[str, float], dict[str, int]]]:
        """
        Evaluates the model on a test dataset.

        Parameters
        ----------
        test_data : list
            Input features for testing.
        expected_results : list
            Ground truth values.

        Returns
        -------
        dict or tuple
            Evaluation metrics depending on ``work_type``.
        """
        results = {}
        if (self.work_type == "reg"):
            results = self.regreccion_test_accuracy(test_data,expected_results)
        elif(self.work_type == "class"):
            results = self.classification_test_accuracy(test_data,expected_results)
        else:
            raise RuntimeError("Tipo de trabajo especificado no es valido.")
        
        return results

    def classification_test_accuracy(self, test_data: list, expected_results: list)->tuple[dict[str, float], dict[str, int]]:
        """
        Calculates classification metrics (Accuracy, Precision, Recall/TPR, F1 Score).

        Note: Currently assumes binary classification or multi-label where threshold is 0.5.

        Returns
        -------
        tuple[dict[str, float], dict[str, int]]
            A tuple containing:
            
            1.  Dictionary of metrics (Acur, Press, TPR, F1).
            2.  Dictionary of raw counts (TP, TN, FP, FN).
        """
        predictions_raw = self.predict(test_data)
        
        predictions = (predictions_raw > 0.5).astype(int).flatten()
        
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
        
        if results.shape != expected_results.shape:
            raise ValueError(f"Las formas de las predicciones {results.shape} y los resultados esperados {expected_results.shape} no coinciden.")

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
    
    def save_model(self, path: str, model_name: str, description: str = None)->None:
        """
        Saves the model architecture, parameters, optimizer state, and wrapper configuration to a file.

        The file is saved as a compressed NumPy archive (``.npz``).

        Parameters
        ----------
        path : str
            Directory path to save the file.
        model_name : str
            Name of the file (without extension).
        description : str, optional
            Optional description to store in metadata.
        """
        if not path.endswith(".npz"):
            path = os.path.join(path,model_name + ".npz")
        
        try:
            self.model.change_device("CPU")
            architecture_config = self.model.get_config()
            
            metadata = {
                'model_name': model_name,
                'description': description,
                'save_timestamp': datetime.datetime.now().isoformat(),
                'total_training_iterations': self.model.num_complited_train_iterations,
                'total_epochs_iteratios':self.model.num_completed_epochs,
                'normalize_inputs': self.normalize_inputs,
                'normalize_outputs': self.normalize_outputs,
                "loaded_train_data":self._loaded_train_data
            }

            normalization_stats = {
                'x_min': self.x_min,
                'x_max': self.x_max,
                'y_min': self.y_min,
                'y_max': self.y_max
            }
            
            config_to_save = {
                'architecture': architecture_config,
                'metadata': metadata,
                'normalization_stats': normalization_stats,
                'optimizer_state': self.model.UPDATE_METHOD.get_state()
            }

            params = self.model.get_parameters()

            # Flatten the parameters to avoid saving nested dictionaries as object arrays
            flat_params = {}
            for layer_key, layer_params in params.items():
                for param_key, param_value in layer_params.items():
                    flat_params[f"{layer_key}_{param_key}"] = param_value

            np.savez_compressed(path, config=config_to_save, **flat_params)

        except Exception as e:
            raise IOError(f"Error al guardar el modelo.") from e

    def load_model(self,path: str)->None:
        """
        Loads a model from a ``.npz`` file created by :meth:`save_model`.

        Reconstructs the :class:`~HeteroSymNN.Core.Nets.neural_nets.HeteroDense`, restores weights, 
        optimizer state, and normalization statistics.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.
        
        Raises
        ------
        IOError
            If the file cannot be loaded or has an invalid format.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        
        try:
            with np.load(path,allow_pickle=True) as data:         
                config_wrapper = data['config'].item()
                architecture_config = config_wrapper['architecture']
                metadata = config_wrapper.get('metadata', {})

                normalize_inputs_flag = metadata.get('normalize_inputs', True)
                normalize_outputs_flag = metadata.get('normalize_outputs', True)
                self.normalize_inputs =normalize_inputs_flag
                self.normalize_outputs = normalize_outputs_flag
                
                metadata.get('total_training_iterations', 0)
                normalization_stats = config_wrapper.get('normalization_stats', {})

                loss_fn_config = architecture_config['loss_config']
                loss_class_name:str = loss_fn_config.pop('class_name')
                if loss_class_name not in registry.LOSS_FN_MAP:
                    raise IOError(f"Función de pérdida desconocida: {loss_class_name}.")
                loss_fn = registry.LOSS_FN_MAP[loss_class_name](**loss_fn_config) 
                

                optimizer_config = architecture_config['optimizer_config']
                opt_class_name:str = optimizer_config.pop('class_name')
                if opt_class_name not in registry.OPTIMIZER_MAP:
                    raise IOError(f"Optimizador desconocido: {opt_class_name}.")
                optimizer = registry.OPTIMIZER_MAP[opt_class_name](**optimizer_config)

                initializer = None
                if 'initializer_config' in architecture_config:
                    init_config = architecture_config['initializer_config']
                    init_class_name = init_config.pop('class_name', None)
                    if init_class_name and init_class_name in registry.INITIALIZER_MAP:
                        # Instanciamos la clase de inicialización con sus parámetros guardados (si tuviera)
                        initializer = registry.INITIALIZER_MAP[init_class_name](**init_config)
                
                # 3. Recrear la arquitectura
                if (('nodes_structure' in architecture_config) and ('detailed_activations' in architecture_config)):
                    self.model = HeteroDense(
                        nodes_structure=architecture_config['nodes_structure'],
                        detailed_activations=architecture_config['detailed_activations'],
                        initializer=initializer,
                        learning_rate=architecture_config['learning_rate'],
                        learning_mode=architecture_config.get('learning_mode', 'Static'),
                        training_mode=architecture_config.get('training_mode', 'stochastic'),
                        batch_size=architecture_config.get('batch_size', 1),
                        optimizer=optimizer,
                        loss_function=loss_fn,
                        num_treaning_iter=architecture_config.get('num_treaning_iterations', 100)
                    )
                elif architecture_config.get('layers_configuration'):
                    raise IOError("Formato de archivo obsoleto. El modelo usa 'layers_configuration' pero la clase espera 'nodes_structure'.")
                else:
                    raise IOError("Archivo de configuración no reconocido o dañado.")

                # Reconstruct the nested parameter dictionary from the flat structure
                params_dict = {}
                for k, v in data.items():
                    if k == 'config':
                        continue
                    
                    # k is like 'layer_0_weights'
                    # Find the last underscore to split layer_key from param_key
                    key_parts = k.rpartition('_')
                    layer_key = key_parts[0]  # e.g., 'layer_0'
                    param_key = key_parts[2]  # e.g., 'weights'

                    if not layer_key or not param_key:
                        warnings.warn(f"Skipping malformed parameter key '{k}' in model file.")
                        continue

                    if layer_key not in params_dict:
                        params_dict[layer_key] = {}
                    params_dict[layer_key][param_key] = v

                self.model.set_parameters(params_dict)
                
                try:
                    self.model.UPDATE_METHOD.set_state(config_wrapper.get('optimizer_state'), self.model._CALCULATION_MANAGER)
                except Exception as e:
                    if (HW.WARNINGS_STRICT_MODE):
                        raise IOError("No se pudo restaurar el estado del optimizador.") from e
                    else:
                        warnings.warn(f"No se pudo restaurar el estado del optimizador. Causa: {e}. El optimizador se reiniciará.")

                normalization_stats = config_wrapper.get('normalization_stats', {})
                self.model.num_complited_train_iterations = metadata.get('total_training_iterations', 0)
                self.model.num_completed_epochs = metadata.get('total_epochs_iteratios',0)
                
                self._loaded_train_data = metadata.get("loaded_train_data",True)
                if (self._loaded_train_data):
                    if (self.normalize_inputs):
                        self.x_min = normalization_stats.get('x_min')
                        self.x_max = normalization_stats.get('x_max')
                        if self.x_min is None:
                            raise IOError("Error: El modelo requiere normalización de entrada (x_min/x_max) pero no se encontraron en el archivo.")
                    
                    # Cargar stats de Y solo si es necesario
                    if (self.normalize_outputs):
                        self.y_min = normalization_stats.get('y_min')
                        self.y_max = normalization_stats.get('y_max')
                        if self.y_min is None:
                            raise IOError("Error: El modelo requiere normalización de salida (y_min/y_max) pero no se encontraron en el archivo.")

        except Exception as e:
            raise IOError(f"Error al cargar el modelo desde {path}. El archivo no se pudo leer, está dañado o no es un modelo válido. Causa: {e}") from e

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
    def __init__(self,Model_class:Callable,work_type:Literal["class","reg"],validation_testing_split = 0.2,
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
        self.best_model: HeteroDense = None
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
                    higher_is_better: bool = True)->tuple[HeteroDense, dict[str, any], list[dict[str, any]]]:
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
                     batch_size: int = None)->Union[list[float], tuple[HeteroDense, dict[str, any], list[dict[str, any]]]]:
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