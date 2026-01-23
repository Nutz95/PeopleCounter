import tensorrt as trt
import torch # Import torch first to ensure CUDA DLLs are loaded on Windows
import sys
import os

def build_engine(onnx_path, engine_path, max_batch_size=1):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # 2GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.FP16)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # --- CONFIGURATION DU BATCHING DYNAMIQUE ---
    # On définit un profil d'optimisation pour supporter de 1 à max_batch_size images
    if max_batch_size >= 1:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape
        
        # On s'assure que les dimensions dynamiques (-1) sont remplacées par des valeurs fixes dans le profil
        # On part sur du 1080p par défaut si c'est dynamique
        def fix_dims(dims, b):
            new_dims = list(dims)
            new_dims[0] = b # Always set batch
            for i in range(1, len(new_dims)):
                if new_dims[i] <= 0: # If dynamic, provide a default
                    if i == 1: new_dims[i] = 3
                    elif i == 2: new_dims[i] = 640 # Safer default for YOLO
                    elif i == 3: new_dims[i] = 640
                    else: new_dims[i] = 640
            return new_dims

        min_shape = fix_dims(input_shape, 1)
        opt_shape = fix_dims(input_shape, max_batch_size)
        max_shape = fix_dims(input_shape, max_batch_size)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"Optimization profile added for batch range [1, {max_batch_size}] with shapes {opt_shape}")

    print(f"Building Engine ({onnx_path} -> {engine_path})...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build engine.")
        return False
        
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")
    return True

if __name__ == "__main__":
    # Usage: python convert_onnx_to_trt.py model.onnx model.engine [max_batch]
    onnx = sys.argv[1] if len(sys.argv) > 1 else 'yolo12x.onnx'
    engine = sys.argv[2] if len(sys.argv) > 2 else 'yolo12x.engine'
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    if os.path.exists(onnx):
        build_engine(onnx, engine, max_batch_size=batch)
    else:
        print(f"File not found: {onnx}")
