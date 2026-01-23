import sys
import os
from pathlib import Path
sys.path.append(r"E:\AI\lwcc")
from lwcc import get_count_from_frame
from lwcc.util.functions import preprocess_frame
import cv2
import numpy as np
from PIL import Image
import torch

class PeopleCounterProcessor:
    def __init__(self, model_name="CSRNet", model_weights="SHA", backend="torch", openvino_device="GPU"):
        """_summary_

        Args:
            model_name (str, optional): _description_. Defaults to "CSRNet".
            model_weights (str, optional): _description_. Defaults to "SHA".
            possible models: 'CSRNet' (SHA, SHB), 'SFANet' (SHA, SHB),'Bay' (QNRF, SHA, SHB), 'DM-Count' (QNRF, SHA, SHB)
        """
        self.model_name = model_name
        self.model_weights = model_weights
        self.model = None
        self.backend = (backend or "torch").lower()
        self.openvino_device = openvino_device
        self.ov_compiled_model = None
        self.ov_input = None
        self.ov_output = None
        self.ov_input_shape = None
        self.trt_context = None
        self.trt_engine = None
        self.trt_inputs = []
        self.trt_outputs = []
        self.trt_bindings = []
        self.cudart = None

        if self.backend == "tensorrt":
            engine_path = os.environ.get("LWCC_TRT_ENGINE", "dm_count.engine")
            if os.path.isfile(engine_path):
                try:
                    import tensorrt as trt
                    from cuda.bindings import runtime as cudart
                    self.cudart = cudart
                    logger = trt.Logger(trt.Logger.INFO)
                    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                        self.trt_engine = runtime.deserialize_cuda_engine(f.read())
                    print(f"LWCC backend loaded into TensorRT: {engine_path}")
                    
                    err, self.trt_stream = cudart.cudaStreamCreate()
                    if int(err) != 0:
                        raise RuntimeError(f"CUDA Stream Creation failed: {err}")
                        
                    self.trt_context = self.trt_engine.create_execution_context()
                    
                    for i in range(self.trt_engine.num_io_tensors):
                        name = self.trt_engine.get_tensor_name(i)
                        shape = self.trt_engine.get_tensor_shape(name)
                        dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                        size = trt.volume(shape)
                        nbytes = size * np.dtype(dtype).itemsize
                        err, device_ptr = cudart.cudaMalloc(nbytes)
                        host_mem = np.empty(shape, dtype=dtype)
                        binding = {'name': name, 'dtype': dtype, 'shape': shape, 'size': size, 'nbytes': nbytes, 'host': host_mem, 'device': device_ptr}
                        if self.trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                            self.trt_inputs.append(binding)
                            self.trt_input_shape = shape
                        else:
                            self.trt_outputs.append(binding)
                        self.trt_context.set_tensor_address(name, int(device_ptr))
                    print(f"PeopleCounterProcessor using TensorRT on RTX: {engine_path}")
                except Exception as exc:
                    print(f"[WARN] TensorRT init failed ({exc}), falling back to openvino.")
                    self.backend = "openvino"
            else:
                self.backend = "openvino"

        if self.backend == "openvino":
            xml_path = os.path.join(str(Path.home()), ".lwcc", "openvino", f"{model_name}_{model_weights}.xml")
            if os.path.isfile(xml_path):
                try:
                    import openvino as ov
                    core = ov.Core()
                    ov_model = core.read_model(xml_path)
                    self.ov_compiled_model = core.compile_model(ov_model, self.openvino_device)
                    self.ov_input = self.ov_compiled_model.inputs[0]
                    self.ov_output = self.ov_compiled_model.outputs[0]
                    try:
                        self.ov_input_shape = list(self.ov_input.get_shape())
                    except Exception:
                        self.ov_input_shape = None
                    print(f"PeopleCounterProcessor using OpenVINO on {self.openvino_device}: {xml_path}")
                except Exception as exc:
                    print(f"[WARN] OpenVINO init failed ({exc}), falling back to torch.")
                    self.backend = "torch"
            else:
                print(f"[WARN] OpenVINO IR not found at {xml_path}, falling back to torch.")
                self.backend = "torch"

        if self.backend != "openvino":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"PeopleCounterProcessor running on device: {device_str}")

    def process(self, frame):
        # Compte les personnes sur la frame
        if self.backend == "tensorrt" and self.trt_context is not None:
            if self.trt_input_shape and len(self.trt_input_shape) == 4:
                _, _, target_h, target_w = self.trt_input_shape
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
            img = preprocess_frame(frame, self.model_name, is_gray=False, resize_img=False)
            if hasattr(img, "numpy"):
                img = img.numpy()
            
            # Copy to device
            input_data = np.ascontiguousarray(img)
            self.cudart.cudaMemcpyAsync(self.trt_inputs[0]['device'], input_data.ctypes.data, self.trt_inputs[0]['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.trt_stream)
            # Execute
            self.trt_context.execute_async_v3(self.trt_stream)
            self.cudart.cudaStreamSynchronize(self.trt_stream)
            # Copy back
            for output in self.trt_outputs:
                self.cudart.cudaMemcpyAsync(output['host'].ctypes.data, output['device'], output['nbytes'], self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.trt_stream)
            
            self.cudart.cudaStreamSynchronize(self.trt_stream)
            
            result = self.trt_outputs[0]['host']
            density = result[0, 0, :, :]
            count = float(np.sum(result))
        elif self.backend == "openvino" and self.ov_compiled_model is not None:
            if self.ov_input_shape and len(self.ov_input_shape) == 4:
                _, _, target_h, target_w = self.ov_input_shape
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
            img = preprocess_frame(frame, self.model_name, is_gray=False, resize_img=False)
            if hasattr(img, "numpy"):
                img = img.numpy()
            result = self.ov_compiled_model([img])[self.ov_output]
            density = result[0, 0, :, :]
            count = float(np.sum(result))
        else:
            count, density = get_count_from_frame(
                frame,
                model_name=self.model_name,
                model_weights=self.model_weights,
                model=self.model,
                return_density=True,
                use_gpu=torch.cuda.is_available()
            )
            self.model = self.model or None

        # --- Optimisation de la visibilité de la carte de densité ---
        # 1. Normalisation dynamique pour utiliser toute la plage de couleurs
        density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 2. Redimensionnement à la taille de la frame
        density_norm = cv2.resize(density_norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 3. Application d'une colormap (JET)
        density_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        
        # il peut détecter des points. On applique un seuil plus strict sur le masque.
        _, mask = cv2.threshold(density_norm, 40, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        density_color = cv2.bitwise_and(density_color, mask_3ch)

        # Si le compte est trop faible (bruit de fond), on le ramène à zéro
        if count < 0.5:
            count = 0
        
        # On retourne la frame originale, la carte de densité brute (fond noir), 
        # le compte, et le masque pour les contours
        return frame, density_color, int(count), mask
