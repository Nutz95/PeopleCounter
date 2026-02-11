import argparse
import numpy as np
import os
import textwrap


def flatten_trt_output(binding):
    arr = np.asarray(binding['host'])
    if arr.size == 0:
        return arr.reshape(0, 0), 0
    if arr.ndim == 1:
        shape = binding.get('shape', arr.shape)
        arr = arr.reshape(shape)
    if arr.ndim == 2:
        return arr, 1
    feature_dim = arr.shape[-1]
    samples = arr.shape[0]
    per_sample = int(np.prod(arr.shape[1:-1])) if arr.ndim > 2 else 1
    flat = arr.reshape(samples * per_sample, feature_dim)
    return flat, per_sample


def dump_flattened(binding, label):
    flat, count = flatten_trt_output(binding)
    print(f"[DBG] Flattened {label}: shape={flat.shape}, per_sample={count}")
    print(textwrap.indent(np.array2string(flat, max_line_width=120), '  '))


def _parse_shape(shape_str):
    return tuple(int(x) for x in shape_str.split(','))


def main():
    parser = argparse.ArgumentParser(description="Flatten TensorRT output buffers into prediction matrices.")
    parser.add_argument("--npy", type=str, help="Path to the .npy file that contains the binding buffer.")
    parser.add_argument("--shape", type=str, default=None, help="Optional shape (comma separated) to reshape the buffer before flattening.")
    args = parser.parse_args()

    if not args.npy or not os.path.isfile(args.npy):
        parser.error("Provide a valid --npy path")

    buffer = np.load(args.npy)
    binding = {'host': buffer}
    if args.shape:
        binding['shape'] = _parse_shape(args.shape)

    flat, count = flatten_trt_output(binding)
    print(f"Flattened shape: {flat.shape}, predictions per sample: {count}")
    print(flat)


if __name__ == "__main__":
    main()
