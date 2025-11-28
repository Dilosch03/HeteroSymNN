import os
import numpy as np
import math as mth
import itertools as iter
from typing import Literal,Callable
import datetime
import warnings
import inspect
import time


from ..Backend import hardware as HW
from ..Core.Nets.neural_nets import ConfigurableNN
from ..Core import losses, optimizers,initializers

def _build_dynamic_map(module, base_class):
    new_map = {}
    for name, member in inspect.getmembers(module):
        if inspect.isclass(member) and \
           issubclass(member, base_class) and \
           member is not base_class:
            
            new_map[name] = member
    return new_map

LOSS_FN_MAP:dict[str,losses.Loss] = _build_dynamic_map(losses, losses.Loss)
OPTIMIZER_MAP:dict[str,optimizers.Optimizer] = _build_dynamic_map(optimizers, optimizers.Optimizer)
INITIALIZER_MAP:dict[str, initializers.Initializer] = _build_dynamic_map(initializers, initializers.Initializer)

def _normalization(vals: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (vals - min_val) / range_val

def _denormalization(vals: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    return vals * (max_val - min_val) + min_val

class Wraper():
    def __init__(self, model: ConfigurableNN,work_type:Literal["class","reg"],normalize_inputs: bool = True, normalize_outputs: bool = True):
        self._model:ConfigurableNN = None
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

        self.model:ConfigurableNN = model

    @property
    def model(self):
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

    def load_training(self, training_data: list, expected_results: list):
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
            self.x_min = None # No se guardan estadísticas
            self.x_max = None
            X_norm = X_raw # Usar los datos en crudo como entrada

        if self.normalize_outputs:
            self.y_min = np.min(Y_raw, axis=0)
            self.y_max = np.max(Y_raw, axis=0)
            Y_norm = _normalization(Y_raw, self.y_min, self.y_max)
        else:
            self.y_min = None 
            self.y_max = None
            Y_norm = Y_raw # Usar los datos en crudo como objetivo
        
        self.training_data_norm = (X_norm, Y_norm)

    def run_training(self, num_iterations:int = None, training_mode: Literal["batch", "mini-batch", "stochastic"] = None, batch_size: int = None):
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
    
    def predict(self, data: list):
        X_raw = np.array(data)
        
        if self.normalize_inputs:
            if self.x_min is None:
                 raise ValueError("El modelo fue entrenado con normalize_inputs=True pero no se encontraron estadísticas (x_min). ¿Fue cargado correctamente?")
            X_norm = _normalization(X_raw, self.x_min, self.x_max)
        else:
            X_norm = X_raw
            
        Y_pred_norm = self.model.forward(X_norm)
        
        if self.normalize_outputs:
            if self.y_min is None:
                raise ValueError("El modelo fue entrenado con normalize_outputs=True pero no se encontraron estadísticas (y_min). ¿Fue cargado correctamente?")
            Y_denorm = _denormalization(Y_pred_norm, self.y_min, self.y_max)
            return Y_denorm
        else:
            return Y_pred_norm 

    def classification_test_accuracy(self, test_data: list, expected_results: list):
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
    
    def regreccion_test_accuracy(self, test_data: list, expected_results: list):
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
    
    def save_model(self, path: str, model_name: str, description: str = None):
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
            
            np.savez_compressed(path, config=config_to_save, **params)
            

        except Exception as e:
            raise IOError(f"Error al guardar el modelo.") from e

    def load_model(self,path: str):
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
                if loss_class_name not in LOSS_FN_MAP:
                    raise IOError(f"Función de pérdida desconocida: {loss_class_name}.")
                loss_fn = LOSS_FN_MAP[loss_class_name](**loss_fn_config) 
                

                optimizer_config = architecture_config['optimizer_config']
                opt_class_name:str = optimizer_config.pop('class_name')
                if opt_class_name not in OPTIMIZER_MAP:
                    raise IOError(f"Optimizador desconocido: {opt_class_name}.")
                optimizer = OPTIMIZER_MAP[opt_class_name](**optimizer_config)

                initializer = None
                if 'initializer_config' in architecture_config:
                    init_config = architecture_config['initializer_config']
                    init_class_name = init_config.pop('class_name', None)
                    if init_class_name and init_class_name in INITIALIZER_MAP:
                        # Instanciamos la clase de inicialización con sus parámetros guardados (si tuviera)
                        initializer = INITIALIZER_MAP[init_class_name](**init_config)
                
                # 3. Recrear la arquitectura
                if (('nodes_structure' in architecture_config) and ('detailed_activations' in architecture_config)):
                    self.model = ConfigurableNN(
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

                params_dict = {k: v for k, v in data.items() if k != 'config'}
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

class GridSearchWraper(Wraper):
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
        self.best_model: ConfigurableNN = None
        self.best_params: dict[str, any] = None
        self.best_score: float = -np.inf
        self.grid_search_results: list[dict[str, any]] = []

    def load_training(self, training_data: list, expected_results: list,shuffle:bool = True):
        """
        Sobrescribe load_training para barajar y partir los datos en
        conjuntos de entrenamiento y validación.
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
        if not param_grid:
            return []

        param_keys = param_grid.keys()
        value_lists = param_grid.values()

        # itertools.product hace todo el trabajo
        combinations_list = list(iter.product(*value_lists))

        # Convierte la lista de tuplas de vuelta a lista de diccionarios
        combinations_dict = [dict(zip(param_keys, combo)) for combo in combinations_list]

        return combinations_dict

    def _run_grid_search(self,
                    static_params: dict[str, any],
                    param_grid: dict[str, list[any]],
                    metric_to_optimize: str = None,
                    higher_is_better: bool = True):
        
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
                    metrics = self.regreccion_test_accuracy(self._X_vali, self._y_vali, num_features=num_features)
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

        # Al final, asigna el mejor modelo al wrapper
        self.model = self.best_model
        
        # Imprime los mejores parámetros de forma legible
        return self.best_model, self.best_params, self.grid_search_results

    # --- 2. NUEVA FUNCIÓN run_training (SOBRESCRITA) ---
    def run_training(self,
                     static_params: dict[str, any] = None,
                     param_grid: dict[str, list[any]] = None,
                     metric_to_optimize: str = 'R2',
                     higher_is_better: bool = True,
                     
                     # --- Args para Entrenamiento Normal (de la clase base) ---
                     num_iterations: int = None, 
                     training_mode: Literal["batch", "mini-batch", "stochastic"] = None, 
                     batch_size: int = None):
        """
        Sobrescribe run_training para manejar dos casos:
        1. Si se provee un 'param_grid', ejecuta un grid search.
        2. Si no, y ya existe un modelo, lo entrena más (llama a super()).
        """
        
        # --- Decidimos qué acción tomar ---
        
        if param_grid is not None:
            # --- Caso 1: El usuario quiere hacer un Grid Search ---
            print("Grid search detectado (param_grid no es None). Iniciando búsqueda...")
            if ((self._X_train is None) or (self._y_train is None) or (self._X_vali is None) or (self._y_vali is None) or (self.model_class is None) or (static_params is None)):
                raise ValueError("Para grid search, debe proveer X_train, y_train, X_test, y_test, model_class, y static_params.")
            
            # Llamamos a la lógica de grid search (la función renombrada)
            return self._run_grid_search(
                static_params=static_params,
                param_grid=param_grid,
                metric_to_optimize=metric_to_optimize,
                higher_is_better=higher_is_better
            )
            
        elif self.model is not None and num_iterations is not None:
            # --- Caso 2: El usuario quiere entrenar más el modelo existente ---
            print(f"Entrenando modelo existente por {num_iterations} iteraciones más...")
            
            # Llamamos a la lógica de la clase base (Wraper)
            return super().run_training(
                num_iterations=num_iterations,
                training_mode=training_mode,
                batch_size=batch_size
            )
        
        elif self.model is None and num_iterations is not None:
            # --- Caso 3: Error - intentando entrenar un modelo que no existe ---
            raise ValueError("No se ha encontrado un modelo. Debe ejecutar run_training con un 'param_grid' primero para encontrar un modelo.")
        
        else:
            # --- Caso 4: Error - no se proveyeron argumentos suficientes ---
            raise ValueError("Argumentos insuficientes. Debe proveer un 'param_grid' (para grid search) o 'num_iterations' (para un modelo existente).")