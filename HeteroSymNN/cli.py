import argparse
import sys
import os
import importlib.util

def _load_custom_scripts(scripts_paths: list[str]):
    for path in scripts_paths:
        if not os.path.exists(path):
            print(f"Warning: Custom script not found at {path}")
            continue
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            print(f"Loaded custom script: {path}")

def main() -> int:
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="HeteroSymNN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    #Hardware
    parser_hardware = subparsers.add_parser("hardware", help="Get the detected harware of the system.")

    #Defaults
    parser_defaults = subparsers.add_parser("defaults", help="Manage framework defaults and cache.")
    parser_defaults.add_argument("--clear-cpu-cache", action="store_true", help="Clear the CPU JIT cache.")
    parser_defaults.add_argument("--show-gpu-cache", action="store_true", help="Returns the path to the GPU JIT cache.")
    parser_defaults.add_argument(
        "--set", 
        type=str, 
        choices=["compute", "cpu-cache", "warnings", "threads", "use-cache"],
        help=(
            "Set a default property. Accepted values depend on the property:\n"
            "  compute   -> GPU_CUDA, CPU_JIT, CPU_PYTHON\n"
            "  cpu-cache -> <path/to/directory>\n"
            "  warnings  -> error, ignore, always, default, module, once\n"
            "  threads   -> <integer> (e.g., 4, 8)\n"
            "  use-cache -> true, false"
        )
    ) 
    parser_defaults.add_argument("--value", type=str, help="The value for the property specified by --set (e.g., GPU_CUDA).")

    #string parsing
    parser_parse = subparsers.add_parser("parse", help="Test symbolic math strings through the JIT compiler.")
    parser_parse.add_argument("expressions", type=str, nargs="+", help="One or more math strings to validate (e.g., 'sin(x)*alpha' 'tanh(x)').")
    parser_parse.add_argument("--show-parsed", action="store_true", help="Display the formula as interpreted by SymPy.")
    parser_parse.add_argument("--show-derivative", action="store_true", help="Display the calculated symbolic derivative.")

    #Clone
    parser_clone = subparsers.add_parser("clone", help="Clone a .symnn topology and reset its trained weights.")
    parser_clone.add_argument("source", type=str, help="Path to the source .symnn model.")
    parser_clone.add_argument("out", type=str, help="Filename for the newly cloned model.")
    parser_clone.add_argument("--name", type=str, help="New name for the cloned model architecture.")
    parser_clone.add_argument("--description", type=str, help="New description for the cloned model architecture.")
    parser_clone.add_argument("--custom-scripts", type=str, nargs="+", help="Paths to Python scripts containing custom objects to register before cloning.")

    #Inspect
    parser_inspect = subparsers.add_parser("inspect", help="Read and summarize the topology of a .symnn file.")
    parser_inspect.add_argument("model", type=str, help="Path to the .symnn file to inspect.")

    args = parser.parse_args()

    if (args.command == "hardware"):
        from .config import settings
        from .Backend import hardware as HW

        print("Default Compute Method:",settings.default_compute_method)
        print("Available Compute Methods:",settings.available_methods)
        print("Default used CPU threads:",settings.n_jobs)
        print("Detected GPUs:",HW.NUM_GPUS)
        print("Detected CPUs threads:",HW.NUM_CPU_THREADS)
    elif (args.command == "defaults"):
        from .config import settings
        if (args.clear_cpu_cache):
            print("Cleaning HeteroSymNN CPU JIT cache...")
            settings.clear_kernel_cache(cache_type="CPU")
            print("CPU Cache cleared successfully.")
        elif (args.show_gpu_cache):
            import os
            home_dir = os.path.expanduser('~')
            cupy_cache_dir = os.path.join(home_dir, '.cupy', 'kernel_cache')
            print(f"GPU (CuPy) Cache Directory: {cupy_cache_dir}")
            print("Note: HeteroSymNN does not clear this automatically due to permission safeguards.")
        elif (args.set):
            if not (args.value):
                print("Error: Please provide a value using --value (e.g., --value GPU_CUDA or CPU_PYTHON)")
            else:
                try:
                    if args.set == "compute":
                        settings.set_default_compute_method(args.value)
                    elif args.set == "cpu-cache":
                        settings.set_cache_location(args.value)
                    elif args.set == "warnings":
                        settings.set_warning_level(args.value)
                    elif args.set == "threads":
                        settings.n_jobs = int(args.value)
                    elif args.set == "use-cache":
                        # Convert common true/false strings to actual python booleans
                        settings.use_kernel_cache = args.value.lower() in ['true', '1', 't', 'y', 'yes']
                    settings.save()
                    print(f"Global property '{args.set}' successfully set to: {args.value}")
                except ValueError as e:
                    print(f"Invalid value for '{args.set}': {e}")
                except Exception as e:
                    print(f"Error setting '{args.set}': {e}")
        else:
            print(f"Global Configuration File: Not Persistent")
            print(f"Compute Method:  {settings.default_compute_method}")
            print(f"Threads:         {settings.n_jobs}")
            print(f"Use Cache:       {settings.use_kernel_cache}")
            print(f"Warning Level:   {settings.warning_level}")
    elif args.command == "parse":
        from .JIT.compiler import SymbolicJITCompiler
        motor = SymbolicJITCompiler([("relu",{})],"CPU_PYTHON",0)
        for expr in args.expressions:
            try:
                func_expr, deriv_expr, required_constants = motor._get_ccode_from_config(expr)
                if (args.show_parsed):
                    print("Function Expression:",func_expr)
                if (args.show_derivative):
                    print("Derivative Expression:",deriv_expr)
                if required_constants:
                    print("Detected Constants:", ", ".join(sorted(list(required_constants))))
                else:
                    print("Detected Constants: None")
                print()
                
            except Exception as e:
                print("Error parsing expression:",e)
    elif args.command == "clone":
        if args.custom_scripts:
            _load_custom_scripts(args.custom_scripts)
        print(f"Cloning topology from {args.source} into {args.out}...")
        try:
            import zipfile
            import json
            from HeteroSymNN.API.wrappers import Wrapper
            from HeteroSymNN.API import registries
            
            with zipfile.ZipFile(args.source, 'r') as archive:
                config_bytes = archive.read("config.json")
                config_wrapper = json.loads(config_bytes)
                
            metadata = config_wrapper.get('metadata', {})
            model_class_name = metadata.get('model_class')
            if model_class_name in registries.registry.legacy_map:
                model_class_name = registries.registry.legacy_map[model_class_name]
                
            TargetNetworkClass = registries.registry._net_map[model_class_name]
            
            # Construct model structure with fresh random weights
            model = TargetNetworkClass.from_config(config_wrapper, registry_module=registries.registry)
            model.num_completed_train_iterations = 0
            model.num_completed_epochs = 0
            model._UPDATE_METHOD._initialize_state(model.layers)
            
            # Setup data transformers if available
            transformation_configs = config_wrapper.get('transformation_configs', {})
            
            input_transformer = None
            if "inputs" in transformation_configs and transformation_configs["inputs"] is not None:
                if metadata.get("input_transformer"):
                    input_transformer = registries.registry.data_transformers_map[metadata["input_transformer"]]()
                    input_transformer.set_config(transformation_configs["inputs"])
                    
            output_transformer = None
            if "outputs" in transformation_configs and transformation_configs["outputs"] is not None:
                if metadata.get("output_transformer"):
                    output_transformer = registries.registry.data_transformers_map[metadata["output_transformer"]]()
                    output_transformer.set_config(transformation_configs["outputs"])
            
            # Create wrapper to save
            wrapper = Wrapper(
                model=model, 
                work_type=metadata.get('work_type'),
                input_transformer=input_transformer,
                output_transformer=output_transformer
            )
            
            name = args.name if args.name else metadata.get('model_name')
            
            wrapper.save_model(args.out, model_name=name, description=args.description, overwrite=True)
            print(f"Successfully cloned model to {args.out}")
        except Exception as e:
            print(f"Error cloning model: {e}")


    elif args.command == "inspect":
        import zipfile
        import json

        print(f"Inspecting internal structure of {args.model}...\n")
        
        if not args.model.endswith(".symnn"):
            args.model += ".symnn"
            
        try:
            with zipfile.ZipFile(args.model, 'r') as archive:
                config_bytes = archive.read("config.json")
                config_wrapper = json.loads(config_bytes)
                
            metadata = config_wrapper.get('metadata', {})
            model_class = metadata.get('model_class', 'Unknown')
            model_name = metadata.get('model_name', 'Unnamed Model')
            desc = metadata.get('description', 'No description provided')
            framework_ver = metadata.get('framework_version', 'Unknown')
            timestamp = metadata.get('save_timestamp', 'Unknown')
            task_type = metadata.get('work_type', 'Unknown')

            print("="*50)
            print(" METADATA ")
            print("="*50)
            print(f"Name:        {model_name}")
            print(f"Description: {desc}")
            print(f"Saved At:    {timestamp}")
            print(f"Framework:   v{framework_ver}")
            print(f"Task Type:   {str(task_type).upper()}")
            print()

            print("="*50)
            print(" TOPOLOGY ")
            print("="*50)
            print(f"Architecture Class: {model_class}")
            
            layer_configs = config_wrapper.get('architecture', {}).get('layer_configs', {})
            if layer_configs:
                for idx, (layer_key, layer_config) in enumerate(layer_configs.items()):
                    layer_type_str = ""
                    # For generic networks, print the specific layer type if available
                    if model_class.lower() == "basenetwork":
                        layer_type = layer_config.get("layer_type", "Layer")
                        layer_type_str = f" [{layer_type}]"
                    
                    input_size = layer_config.get('num_inputs', '?')
                    layer_size = layer_config.get('num_nodes', '?')
                    
                    activations = layer_config.get('layer_node_configs', [])
                    if isinstance(activations, list):
                        # Heterogeneous or just multiple
                        from collections import Counter
                        act_strings = []
                        for act in activations:
                            if isinstance(act, str):
                                act_strings.append(act)
                            elif isinstance(act, list) and len(act) == 2:
                                act_strings.append(act[0])
                        
                        counts = Counter(act_strings)
                        act_summary = ", ".join(f"{cnt}x '{func}'" for func, cnt in counts.items())
                        print(f"Layer {idx}{layer_type_str}: {input_size} -> {layer_size}  | Activations: {act_summary}")
                    else:
                        print(f"Layer {idx}{layer_type_str}: {input_size} -> {layer_size}  | Activations: {activations}")
            else:
                print("No layer configuration found.")
            print()

            print("="*50)
            print(" DYNAMIC CONSTANTS ")
            print("="*50)
            has_constants = False
            for idx, (layer_key, layer_config) in enumerate(layer_configs.items()):
                activations = layer_config.get('layer_node_configs', [])
                layer_constants = {} # Map constant_name -> set of unique values
                if isinstance(activations, list):
                    for act in activations:
                        if isinstance(act, list) and len(act) == 2 and isinstance(act[1], dict):
                            for k, v in act[1].items():
                                if k not in layer_constants:
                                    layer_constants[k] = set()
                                layer_constants[k].add(v)
                
                if layer_constants:
                    has_constants = True
                    const_strs = []
                    for k, vals in layer_constants.items():
                        if len(vals) == 1:
                            const_strs.append(f"{k}: {vals.pop()}")
                        elif len(vals) <= 3:
                            # Show up to 3 unique values
                            const_strs.append(f"{k}: {sorted(list(vals))}")
                        else:
                            # If many unique values, show min and max
                            const_strs.append(f"{k}: {len(vals)} unique vals [{min(vals):.4f} .. {max(vals):.4f}]")
                            
                    const_str = ", ".join(const_strs)
                    print(f"Layer {idx}: {const_str}")
            
            if not has_constants:
                print("No dynamic constants detected.")
            print()

            print("="*50)
            print(" PRE-PROCESSING ")
            print("="*50)
            input_trans = metadata.get('input_transformer', None)
            output_trans = metadata.get('output_transformer', None)
            print(f"Input Transformer:  {input_trans if input_trans else 'None'}")
            print(f"Output Transformer: {output_trans if output_trans else 'None'}")
            print("="*50)

        except Exception as e:
            print(f"Error inspecting model: {e}")

    return 0