import os


def dump_trt_bindings(engine, label):
    flag = os.environ.get("PEOPLE_COUNTER_DUMP_TRT_BINDINGS", "0").lower()
    if flag not in ("1", "true", "yes"):
        return
    print(f"[DBG] {label} TensorRT bindings:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"   {mode.name} {name}: shape={shape}, dtype={dtype}")


def dump_trt_allocation_info(engine, label):
    flag = os.environ.get("PEOPLE_COUNTER_DUMP_TRT_BINDINGS", "0").lower()
    if flag not in ("1", "true", "yes"):
        return
    print(f"[DBG] {label} TensorRT allocation info:")
    num_profiles = getattr(engine, 'num_optimization_profiles', 0)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        try:
            runtime_shape = engine.get_tensor_shape(name)
        except Exception:
            runtime_shape = None
        print(f"  Binding {i} name={name} runtime_shape={runtime_shape}")
        if num_profiles:
            for p in range(num_profiles):
                try:
                    p_shape = engine.get_tensor_profile_shape(name, p)
                    print(f"    profile[{p}] = {p_shape}")
                except Exception as e:
                    print(f"    profile[{p}] lookup failed: {e}")
        else:
            print("    no optimization profiles reported")
