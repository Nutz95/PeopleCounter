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
        if self.backend == "openvino" and self.ov_compiled_model is not None:
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
        # Redimensionner la carte de densité à la taille de la frame
        density_img = Image.fromarray(np.uint8(density * 255), 'L')
        density_img = density_img.resize((frame.shape[1], frame.shape[0]), Image.BILINEAR)
        density_np = np.array(density_img.convert('RGB'))
        # Appliquer une colormap rouge (COLORMAP_JET donne du rouge pour les valeurs hautes)
        density_color = cv2.applyColorMap(density_np, cv2.COLORMAP_JET)

        # Superposer la carte de densité sur la frame
        overlay = cv2.addWeighted(frame, 0.7, density_color, 0.3, 0)
        # (Suppression de l'annotation 'Count:' en rouge)
        return frame, overlay, int(count)
