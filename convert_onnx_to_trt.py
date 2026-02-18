import tensorrt as trt
import sys
import os
from pathlib import Path

try:
    from app_v2.infrastructure.timing_cache_manager import TimingCacheManager as _TCM
except ImportError:
    _TCM = None  # type: ignore[assignment]

try:
    from app_v2.infrastructure.engine_refitter import write_refit_sidecar as _write_sidecar
except ImportError:
    _write_sidecar = None  # type: ignore[assignment]

try:
    from app_v2.infrastructure.engine_stream_writer import FileStreamWriter as _FileStreamWriter
except ImportError:
    _FileStreamWriter = None  # type: ignore[assignment]


def build_engine(onnx_path, engine_path, max_batch_size=1, strip_plan=False):
    logger = trt.Logger(trt.Logger.INFO)
    try:
        builder = trt.Builder(logger)
    except Exception as e:
        print(f"TensorRT Builder creation failed: {e}")
        print("Make sure CUDA and TensorRT are properly installed and visible to the Python environment.")
        return False
    config = builder.create_builder_config()
    
    # 2GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.FP16)

    # Weight stripping — produces a smaller engine that must be refitted before use
    if strip_plan:
        if hasattr(trt.BuilderFlag, "STRIP_PLAN"):
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
        else:
            print("[WARNING] STRIP_PLAN not supported by this TRT version; building full engine")
            strip_plan = False

    # Timing cache — persists kernel tactic choices for deterministic, faster rebuilds
    _cache_path = Path(engine_path).parent / "timing_cache.bin"
    _tc = _TCM(str(_cache_path)) if _TCM is not None else None
    if _tc is not None:
        _tc.load_into_config(config)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Use absolute path and change dir to handle external weights (.onnx.data)
    onnx_abs = os.path.abspath(onnx_path)
    onnx_dir = os.path.dirname(onnx_abs)
    old_cwd = os.getcwd()
    os.chdir(onnx_dir)
    
    try:
        with open(onnx_abs, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return False
    finally:
        os.chdir(old_cwd)
    
    # --- CONFIGURATION DU BATCHING DYNAMIQUE ---
    # On définit un profil d'optimisation pour supporter de 1 à max_batch_size images
    if max_batch_size >= 1:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        input_shape = network.get_input(0).shape
        
        # On s'assure que les dimensions dynamiques (-1) sont remplacées par des valeurs fixes dans le profil
        # On définit les dimensions optimales : 1080p pour densité si dynamique
        def fix_dims(dims, b, model_name, target_res="opt"):
            new_dims = list(dims)
            new_dims[0] = b # Always set batch
            is_density = any(word in model_name.lower() for word in ["dm_count", "csrnet", "sfanet", "bay", "density"])
            
            for i in range(1, len(new_dims)):
                if new_dims[i] <= 0: # If dynamic, provide a default
                    if i == 1: new_dims[i] = 3
                    elif i == 2:
                        # Pour la densité, on force maintenant la calibration 720x640 demandée pour les tiles
                        if is_density:
                            new_dims[i] = 720
                        else:
                            new_dims[i] = 640
                    elif i == 3:
                        if is_density:
                            new_dims[i] = 640
                        else:
                            new_dims[i] = 640
                    else: 
                        new_dims[i] = 640
            return new_dims

        model_filename = os.path.basename(onnx_path)
        min_shape = fix_dims(input_shape, 1, model_filename, "min")
        opt_shape = fix_dims(input_shape, max_batch_size, model_filename, "opt")
        max_shape = fix_dims(input_shape, max_batch_size, model_filename, "opt")
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"Optimization profile added for batch range [1, {max_batch_size}] with shapes {opt_shape}")

    print(f"Building Engine ({onnx_path} -> {engine_path})...")
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)

    # Prefer streaming serialisation to avoid holding the full engine in RAM
    _can_stream = (
        _FileStreamWriter is not None
        and hasattr(builder, "build_serialized_network_to_stream")
    )
    if _can_stream:
        writer = _FileStreamWriter(engine_path)
        ok = builder.build_serialized_network_to_stream(network, config, writer)
        writer.close()
        if not ok:
            print("ERROR: build_serialized_network_to_stream failed.")
            return False
        print(f"Engine saved to {engine_path} ({writer.bytes_written / (1024**2):.1f} MB) [stream]")
    else:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("ERROR: Failed to build engine.")
            return False
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_path}")

    # Persist timing cache after successful build
    if _tc is not None:
        _tc.save_from_config(config)

    # Write sidecar so EngineRefitter can locate the ONNX without extra args
    if strip_plan and _write_sidecar is not None:
        _write_sidecar(engine_path, os.path.abspath(onnx_path))
        print(f"[WeightStrip] Sidecar written: {engine_path}.onnxpath")

    return True

if __name__ == "__main__":
    # Usage: python convert_onnx_to_trt.py model.onnx model.engine [max_batch]
    onnx = sys.argv[1] if len(sys.argv) > 1 else 'yolo26x.onnx'
    engine = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x.engine'
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    strip = "--strip" in sys.argv

    if os.path.exists(onnx):
        build_engine(onnx, engine, max_batch_size=batch, strip_plan=strip)
    else:
        print(f"File not found: {onnx}")
