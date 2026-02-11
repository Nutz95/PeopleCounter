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
