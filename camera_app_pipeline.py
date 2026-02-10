import base64
import cv2
import json
import sys
import screeninfo
import time
import csv
import os
import threading
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from camera_capture import CameraCapture
from rtsp_capture import RTSPCapture
from mqtt_capture import MqttCapture
from pipeline_components import MetricsCollector, ModelLoader, ProfileManager

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from cuda_pipeline.preview import CudaPreviewComposer
except Exception:
    CudaPreviewComposer = None

# --- Task for threading models ---
class ModelThread(threading.Thread):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.duration = 0
    
    def run(self):
        start = time.time()
        self.result = self.func(*self.args, **self.kwargs)
        self.duration = time.time() - start

# Période de moyennage en secondes
AVERAGING_PERIOD_SECONDS = 2  # Réduit à 2s pour un graphe plus réactif

class CameraAppPipeline:
    def __init__(self, capture_mode=None, frame_callback=None, metrics_callback=None):
        cmode = (capture_mode or os.environ.get('CAPTURE_MODE', 'usb')).lower()
        self.capture_mode = cmode
        self.frame_callback = frame_callback
        self.metrics_callback = metrics_callback
        self._cuda_preview_composer = None
        self._cuda_preview_enabled = bool(CudaPreviewComposer is not None and torch is not None and torch.cuda.is_available())
        self.use_gui = os.environ.get('USE_GUI', '0') == '1'
        self.graph_overlay_enabled = os.environ.get('GRAPH_OVERLAY', '1') == '1'

        cam_index = int(os.environ.get('CAMERA_INDEX', '0'))
        cam_width = int(os.environ.get('CAMERA_WIDTH', '1920'))
        cam_height = int(os.environ.get('CAMERA_HEIGHT', '1080'))
        broker_addr = os.environ.get('MQTT_BROKER', '127.0.0.1')
        broker_port = int(os.environ.get('MQTT_PORT', '1883'))
        mqtt_exposure = int(os.environ.get('MQTT_EXPOSURE', '500'))
        self.yolo_conf = float(os.environ.get('YOLO_CONF', '0.65'))
        self.density_threshold = int(os.environ.get('DENSITY_THRESHOLD', '15'))
        self.denoise_strength = int(os.environ.get('DENOISE_STRENGTH', '0'))
        self.yolo_tiling = os.environ.get('YOLO_TILING', '1') == '1'
        self.density_tiling = os.environ.get('DENSITY_TILING', '1') == '1'
        self.debug_tiling = os.environ.get('DEBUG_TILING', '0') == '1'
        self.extreme_debug = os.environ.get('EXTREME_DEBUG', '0') == '1'
        self.yolo_global_tile_only = os.environ.get('YOLO_GLOBAL_TILE_ONLY', '0') == '1'
        self.density_enabled = os.environ.get('DISABLE_DENSITY_INFERENCE', '0') != '1'

        if cmode == 'mqtt':
            print("Using MqttCapture (Maixcam/MQTT)")
            self.capture = MqttCapture(broker_address=broker_addr, broker_port=broker_port, exposure=mqtt_exposure, width=cam_width, height=cam_height)
        elif cmode == 'rtsp':
            rtsp_url = os.environ.get('RTSP_URL', 'rtsp://192.168.1.95:8554/live')
            print(f"Using RTSPCapture ({rtsp_url})")
            self.capture = RTSPCapture(rtsp_url, width=cam_width, height=cam_height)
        else:
            print(f"Using CameraCapture (index={cam_index}, {cam_width}x{cam_height})")
            self.capture = CameraCapture(camera_index=cam_index, width=cam_width, height=cam_height, auto_exposure=1)

        self.profile_manager = ProfileManager()
        self.profile_manager.ensure_loaded()
        self.active_profile_name = self.profile_manager.active_profile_name or ''
        self.available_profiles = list(self.profile_manager.available_profiles)
        self.model_loader = ModelLoader(yolo_conf=self.yolo_conf, density_threshold=self.density_threshold)
        self.metrics_collector = MetricsCollector(history_len=600)
        self.profile_active = False
        self.profile_log = []
        self.latest_metrics = {}
        self.history_data = []
        self.history_len = 600
        self.yolo_pipeline_mode = os.environ.get('YOLO_PIPELINE_MODE', 'auto').lower()
        self.density_quads_coords = []
        self.yolo_device = 'cpu'
        self.density_device = 'GPU'
        self.processor = None
        self.yolo_counter = None
        self.latest_pipeline_report = {}
        self._overlay_stats = {
            "compose_ms": None,
            "preview_ms": None,
            "cpu_preview_ms": None,
            "draw_kernel_ms": None,
            "draw_blend_ms": None,
            "draw_convert_ms": None,
            "draw_total_ms": None,
            "draw_returned_tensor": False,
            "renderer_fallback_reason": None,
        }

        self._apply_profile_settings(self.profile_manager.profile_settings)
        if not self._refresh_models(force=True):
            raise RuntimeError(f"[FATAL] Impossible d'initialiser les moteurs pour "
                               f"le profil '{self.active_profile_name or 'Live Metrics'}'.")

        try:
            screen = screeninfo.get_monitors()[0]
            self.screen_width = screen.width
            self.screen_height = screen.height
        except Exception:
            self.screen_width = 1920
            self.screen_height = 1080

        if self.use_gui:
            cv2.namedWindow("People Count", cv2.WINDOW_NORMAL)
            init_w = int(self.screen_width * 0.75)
            init_h = int(self.screen_height * 0.75)
            cv2.resizeWindow("People Count", init_w, init_h)
            cv2.moveWindow("People Count", (self.screen_width - init_w) // 2, (self.screen_height - init_h) // 2)

        self.use_yolo = True
        self._density_payload_empty_logged = False

    def _apply_profile_settings(self, settings):
        self.yolo_tiling = settings.get('YOLO_TILING', '1') == '1'
        self.density_tiling = settings.get('DENSITY_TILING', '1') == '1'
        self.debug_tiling = settings.get('DEBUG_TILING', '0') == '1'
        self.extreme_debug = settings.get('EXTREME_DEBUG', '0') == '1'
        self.yolo_global_tile_only = settings.get('YOLO_GLOBAL_TILE_ONLY', '1' if self.yolo_global_tile_only else '0') == '1'
        disable_density = settings.get('DISABLE_DENSITY_INFERENCE')
        if disable_density is not None:
            self.density_enabled = disable_density != '1'
        self.yolo_conf = float(settings.get('YOLO_CONF', self.yolo_conf))
        self.density_threshold = int(settings.get('DENSITY_THRESHOLD', self.density_threshold))
        self.denoise_strength = int(settings.get('DENOISE_STRENGTH', self.denoise_strength))
        self.model_loader.yolo_conf = self.yolo_conf
        self.model_loader.density_threshold = self.density_threshold

    def _create_cuda_preview_composer(self):
        if not self._cuda_preview_enabled:
            return None
        if self._cuda_preview_composer is not None:
            return self._cuda_preview_composer
        try:
            quality = int(os.environ.get('CUDA_PREVIEW_JPEG_QUALITY', '80'))
        except ValueError:
            quality = 80
        try:
            self._cuda_preview_composer = CudaPreviewComposer(jpeg_quality=quality)
        except EnvironmentError as exc:
            self._cuda_preview_enabled = False
            if self.extreme_debug:
                print(f"[WARN] CUDA preview composer unavailable: {exc}")
            return None
        return self._cuda_preview_composer

    def _generate_cuda_preview_payload(self, overlay_tensor, win_w, win_h):
        composer = self._create_cuda_preview_composer()
        if composer is None:
            return None
        start = time.perf_counter()
        try:
            return composer.compose(overlay_tensor, target_size=(win_h, win_w))
        except Exception as exc:
            if self.extreme_debug:
                print(f"[WARN] CUDA preview compose failed: {exc}")
            return None
        finally:
            self._overlay_stats["preview_ms"] = (time.perf_counter() - start) * 1000

    def _log_cpu_fallback(self, reason):
        if not hasattr(self, '_cpu_fallback_reasons'):
            self._cpu_fallback_reasons = set()
        if reason in self._cpu_fallback_reasons:
            return
        self._cpu_fallback_reasons.add(reason)
        print(f"[CPU FALLBACK] {reason}")

    def _resize_tensor(self, tensor, size, mode="bilinear"):
        if tensor is None or F is None:
            raise RuntimeError("CUDA resize requested without torch functional API.")
        resize_kwargs = {"mode": mode}
        if mode in ("bilinear", "bicubic"):
            resize_kwargs["align_corners"] = False
        return F.interpolate(
            tensor.unsqueeze(0),
            size=size,
            **resize_kwargs,
        ).squeeze(0)

    def _build_density_tensors(self, density_result, render_size, device):
        if torch is None or density_result is None:
            return None, None
        render_h, render_w = render_size
        target_h = max(1, render_h)
        target_w = max(1, render_w)
        color_tensor = torch.zeros((3, target_h, target_w), device=device, dtype=torch.float32)
        mask_tensor = torch.zeros((1, target_h, target_w), device=device, dtype=torch.float32)
        try:
            if self.density_tiling and len(density_result) >= 4:
                _, quad_colors, _, quad_masks = density_result
                mid_h = target_h // 2
                mid_w = target_w // 2
                positions = [
                    (0, mid_h, 0, mid_w),
                    (0, mid_h, mid_w, target_w),
                    (mid_h, target_h, 0, mid_w),
                    (mid_h, target_h, mid_w, target_w),
                ]
                for idx, (color_np, mask_np) in enumerate(zip(quad_colors, quad_masks)):
                    if color_np is None or mask_np is None:
                        continue
                    y0, y1, x0, x1 = positions[idx]
                    chunk_h = max(1, y1 - y0)
                    chunk_w = max(1, x1 - x0)
                    src_color = torch.from_numpy(color_np).permute(2, 0, 1).to(device=device, dtype=torch.float32)
                    src_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device=device, dtype=torch.float32)
                    resized_color = self._resize_tensor(src_color, (chunk_h, chunk_w))
                    resized_mask = self._resize_tensor(src_mask, (chunk_h, chunk_w), mode="nearest")
                    color_tensor[:, y0:y1, x0:x1] = resized_color
                    mask_tensor[:, y0:y1, x0:x1] = resized_mask
            elif len(density_result) >= 4:
                _, dens_raw, _, dens_mask = density_result
                if dens_raw is None or dens_mask is None:
                    return None, None
                src_color = torch.from_numpy(dens_raw).permute(2, 0, 1).to(device=device, dtype=torch.float32)
                src_mask = torch.from_numpy(dens_mask).unsqueeze(0).to(device=device, dtype=torch.float32)
                color_tensor = self._resize_tensor(src_color, (target_h, target_w))
                mask_tensor = self._resize_tensor(src_mask, (target_h, target_w), mode="nearest")
            else:
                return None, None
        except Exception as exc:
            if self.extreme_debug:
                print(f"[WARN] Failed to build density tensors on CUDA: {exc}")
            return None, None
        return color_tensor, mask_tensor

    def _overlay_hotspots(self, overlay, mask_norm, device):
        if mask_norm is None or overlay is None:
            return overlay
        normalized = mask_norm.squeeze(0)
        if normalized.numel() == 0 or normalized.max() <= 0.05:
            return overlay
        candidates = torch.nonzero(normalized > 0.05, as_tuple=False)
        if candidates.numel() == 0:
            return overlay
        values = normalized[candidates[:, 0], candidates[:, 1]]
        limit = min(6, candidates.shape[0])
        topk_vals, topk_idx = torch.topk(values, limit)
        selection = candidates[topk_idx]
        for coord in selection:
            center_y, center_x = int(coord[0].item()), int(coord[1].item())
            overlay = self._blend_hotspot(overlay, center_y, center_x)
        return overlay

    def _blend_hotspot(self, overlay, center_y, center_x, radius=6):
        if overlay is None:
            return overlay
        h, w = overlay.shape[1], overlay.shape[2]
        y0 = max(0, center_y - radius - 1)
        y1 = min(h, center_y + radius + 2)
        x0 = max(0, center_x - radius - 1)
        x1 = min(w, center_x + radius + 2)
        if y0 >= y1 or x0 >= x1:
            return overlay
        rows = torch.arange(y0, y1, device=overlay.device).view(-1, 1)
        cols = torch.arange(x0, x1, device=overlay.device).view(1, -1)
        dist_sq = (rows - center_y) ** 2 + (cols - center_x) ** 2
        mask = (dist_sq <= (radius ** 2)).float().unsqueeze(0)
        if mask.max() == 0:
            return overlay
        alpha = 0.65
        red = torch.tensor([0.0, 0.0, 255.0], dtype=overlay.dtype, device=overlay.device).view(3, 1, 1)
        patch = overlay[:, y0:y1, x0:x1]
        overlay[:, y0:y1, x0:x1] = patch * (1 - alpha * mask) + red * (alpha * mask)
        return overlay

    def _compose_cuda_overlay(self, frame_tensor, density_result, render_size):
        if torch is None or F is None or not torch.cuda.is_available():
            self._log_cpu_fallback("CUDA overlay skipped: torch CUDA unavailable.")
            return None
        try:
            device = torch.device("cuda")
            start = time.perf_counter()
            base = frame_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
            base_resized = self._resize_tensor(base, render_size)
            density_color, density_mask = self._build_density_tensors(density_result, render_size, device)
            if density_color is None or density_mask is None:
                self._overlay_stats["compose_ms"] = (time.perf_counter() - start) * 1000
                return base_resized
            mask_norm = (density_mask / 255.0).clamp(0.0, 1.0)
            heatmap = density_color * mask_norm
            overlay = torch.clamp(base_resized + heatmap * 0.6, 0, 255)
            overlay = self._overlay_hotspots(overlay, mask_norm, device)
            self._overlay_stats["compose_ms"] = (time.perf_counter() - start) * 1000
            return overlay
        except Exception as exc:
            self._log_cpu_fallback(f"CUDA overlay composition failed: {exc}")
            self._overlay_stats["compose_ms"] = (time.perf_counter() - start) * 1000 if 'start' in locals() else None
            return None

    def _tensor_to_preview_frame(self, tensor):
        if torch is None or tensor is None:
            return None
        if not torch.is_tensor(tensor):
            raise RuntimeError("Expected a torch tensor for CUDA preview conversion.")
        start = time.perf_counter()
        try:
            return tensor.detach().permute(1, 2, 0).contiguous().cpu().numpy()
        finally:
            self._overlay_stats["preview_ms"] = (time.perf_counter() - start) * 1000
            self._overlay_stats["cpu_preview_ms"] = None

    def _get_density_count(self, density_result):
        if not density_result:
            return 0.0
        total = density_result.get('total_count')
        if total is not None:
            return float(total)
        tiles = density_result.get('tiles', [])
        return sum(float(tile.get('count', 0.0)) for tile in tiles)

    def _encode_density_mask(self, mask):
        if mask is None:
            return None
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.size == 0:
            return None
        success, encoded = cv2.imencode('.png', mask_arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            return None
        return base64.b64encode(encoded.tobytes()).decode('ascii')

    def _build_density_heatmap_payload(self, density_result, frame_shape):
        if not density_result:
            return None
        tiles = density_result.get('tiles') or []
        if not tiles:
            return None
        frame_h, frame_w = frame_shape
        quad_coords = getattr(self, 'density_quads_coords', []) or []
        created_ts = time.time()
        created_iso = datetime.utcfromtimestamp(created_ts).isoformat() + "Z"
        payload_tiles = []
        for idx, tile in enumerate(tiles):
            mask = tile.get('mask')
            if mask is None:
                continue
            mask_arr = np.asarray(mask, dtype=np.uint8)
            if mask_arr.size == 0:
                continue
            blob = self._encode_density_mask(mask_arr)
            if not blob:
                continue
            coords = quad_coords[idx] if idx < len(quad_coords) else (0, 0, frame_w, frame_h)
            x0, y0, x1, y1 = coords
            mask_h, mask_w = mask_arr.shape[:2]
            payload_tiles.append({
                'blob': blob,
                'format': 'png',
                'coords': [int(x0), int(y0), int(x1), int(y1)],
                'mask_width': int(tile.get('mask_width', mask_w)),
                'mask_height': int(tile.get('mask_height', mask_h)),
                'tile_width': int(tile.get('tile_width', x1 - x0)),
                'tile_height': int(tile.get('tile_height', y1 - y0)),
                'scale_x': float(tile.get('scale_x', 1.0)),
                'scale_y': float(tile.get('scale_y', 1.0)),
                'count': float(tile.get('count', 0.0)),
            })
        if not payload_tiles:
            return None
        return {
            'created_at': created_iso,
            'created_at_ts': created_ts,
            'frame_width': frame_w,
            'frame_height': frame_h,
            'total_count': float(density_result.get('total_count', 0.0)),
            'tiles': payload_tiles,
        }

    def _cpu_preview_render(self, frame_image, density_result, render_size, win_size):
        self._log_cpu_fallback("CPU preview rendering (CUDA overlay unavailable).")
        start = time.perf_counter()
        render_h, render_w = render_size
        win_h, win_w = win_size
        frame_render = cv2.resize(frame_image, (render_w, render_h))
        density_raw_small = np.zeros((render_h, render_w, 3), dtype=np.uint8)
        density_mask_small = np.zeros((render_h, render_w), dtype=np.uint8)
        mid_h, mid_w = render_h // 2, render_w // 2

        if self.density_tiling and density_result and len(density_result) >= 4:
            _, quad_res_color, _, quad_res_masks = density_result
            density_raw_small[0:mid_h, 0:mid_w] = cv2.resize(quad_res_color[0], (mid_w, mid_h))
            density_raw_small[0:mid_h, mid_w:render_w] = cv2.resize(quad_res_color[1], (render_w-mid_w, mid_h))
            density_raw_small[mid_h:render_h, 0:mid_w] = cv2.resize(quad_res_color[2], (mid_w, render_h-mid_h))
            density_raw_small[mid_h:render_h, mid_w:render_w] = cv2.resize(quad_res_color[3], (render_w-mid_w, render_h-mid_h))

            density_mask_small[0:mid_h, 0:mid_w] = cv2.resize(quad_res_masks[0], (mid_w, mid_h), interpolation=cv2.INTER_NEAREST)
            density_mask_small[0:mid_h, mid_w:render_w] = cv2.resize(quad_res_masks[1], (render_w-mid_w, mid_h), interpolation=cv2.INTER_NEAREST)
            density_mask_small[mid_h:render_h, 0:mid_w] = cv2.resize(quad_res_masks[2], (mid_w, render_h-mid_h), interpolation=cv2.INTER_NEAREST)
            density_mask_small[mid_h:render_h, mid_w:render_w] = cv2.resize(quad_res_masks[3], (render_w-mid_w, render_h-mid_h), interpolation=cv2.INTER_NEAREST)
        elif density_result and len(density_result) >= 4:
            _, dens_raw, _, dens_mask = density_result
            if dens_raw is not None:
                density_raw_small = cv2.resize(dens_raw, (render_w, render_h))
            if dens_mask is not None:
                density_mask_small = cv2.resize(dens_mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

        mask_3ch = cv2.cvtColor(density_mask_small, cv2.COLOR_GRAY2BGR)
        heatmap_on_black = cv2.bitwise_and(density_raw_small, mask_3ch)
        overlay2 = cv2.addWeighted(frame_render, 1.0, heatmap_on_black, 0.6, 0)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        density_mask_dilated = cv2.dilate(density_mask_small, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(density_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cv2.circle(overlay2, (int(x), int(y)), int(radius) + 2, (0, 0, 255), 1)

        scale = min(win_w / render_w, win_h / render_h)
        target_w, target_h = int(render_w * scale), int(render_h * scale)
        preview_frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        resized = cv2.resize(overlay2, (target_w, target_h))
        y_off, x_off = (win_h - target_h) // 2, (win_w - target_w) // 2
        preview_frame[y_off:y_off+target_h, x_off:x_off+target_w] = resized
        self._overlay_stats["cpu_preview_ms"] = (time.perf_counter() - start) * 1000
        self._overlay_stats["preview_ms"] = None
        return preview_frame

    def _refresh_models(self, force=False):
        if not force and getattr(self, '_models_loaded', False):
            return False
        settings = self.profile_manager.profile_settings
        try:
            yolo_counter, processor, metadata = self.model_loader.load(settings)
        except Exception as exc:
            print(f"[WARN] Failed to reload models: {exc}")
            return False
        if yolo_counter is None or processor is None:
            print("[WARN] Model loader returned empty handles, keeping previous engines.")
            if self.density_enabled and processor is None:
                print("[WARN] Density processor unavailable after refresh; density output will stay muted until next reload.")
            return False
        self.yolo_counter = yolo_counter
        self.processor = processor
        self.yolo_device = metadata.get('device', getattr(self.yolo_counter, 'last_device', 'cpu'))
        self.density_device = metadata.get('processor_device', getattr(self.processor, 'last_device', 'GPU'))
        self.model_metadata = metadata
        self._models_loaded = True
        self.set_yolo_pipeline_mode(self.yolo_pipeline_mode)
        if self.yolo_counter and hasattr(self.yolo_counter, 'set_debug_mode'):
            self.yolo_counter.set_debug_mode(self.extreme_debug)
        return True

    def apply_profile(self, profile_name):
        applied = self.profile_manager.apply_profile(profile_name)
        self.active_profile_name = self.profile_manager.pending_profile_name or ''
        return applied

    def set_graph_overlay(self, enabled):
        self.graph_overlay_enabled = bool(enabled)

    def attach_metrics_callback(self, callback):
        self.metrics_callback = callback

    def set_debug_mode(self, enabled):
        self.extreme_debug = bool(enabled)
        if self.yolo_counter and hasattr(self.yolo_counter, 'set_debug_mode'):
            self.yolo_counter.set_debug_mode(self.extreme_debug)

    def set_yolo_pipeline_mode(self, mode):
        normalized = (mode or 'auto').lower()
        if normalized not in ('auto', 'cpu', 'gpu', 'gpu_full'):
            return False
        self.yolo_pipeline_mode = normalized
        if self.yolo_counter and hasattr(self.yolo_counter, 'configure_pipeline'):
            self.yolo_counter.configure_pipeline(normalized)
        return True

    def set_profile_logging(self, active):
        self.profile_active = bool(active)

    def clear_profile_log(self):
        self.profile_log.clear()

    def get_profile_log(self, limit=40):
        return list(self.profile_log[-limit:])

    def run(self):
        # Prépare le fichier CSV avec date et heure
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_filename = f"people_count_{now_str}.csv"
        img_filename = f"people_count_{now_str}.png"
        csv_path = os.path.join(os.getcwd(), csv_filename)
        img_path = os.path.join(os.getcwd(), img_filename)
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["datetime", "YOLO_count", "Density_count", "Avg_combined", "Max_count"])
            counts_yolo = []
            counts_density = []
            last_period = int(time.time())
            log_data = []
            try:
                while True:
                    start_time = time.time()
                    # Retry mechanism for frame acquisition
                    max_retries = 3
                    retry_count = 0
                    frame = self.capture.get_frame()
                    while frame is None and retry_count < max_retries:
                        print(f"[WARNING] Frame is None. Tentative {retry_count+1}/{max_retries}...")
                        time.sleep(1)
                        frame = self.capture.get_frame()
                        retry_count += 1
                    if frame is None:
                        print("[ERROR] Frame is None. Vérifiez la connexion de la caméra et ses paramètres.")
                        print(f"Camera opened: {self.capture.is_opened}")
                        break
                    # print(f"Frame shape: {frame.shape}")
                    if self.profile_manager.pending_profile_reload:
                        forced = self.profile_manager.ensure_loaded()
                        if forced:
                            self._apply_profile_settings(self.profile_manager.profile_settings)
                            if not self._refresh_models(force=True):
                                print("[ERROR] Impossible de recharger les modèles YOLO/Density, arrêt.")
                                break
                    self.active_profile_name = self.profile_manager.active_profile_name or 'Live Metrics'
                    preprocess_duration = 0.0
                    postprocess_duration = 0.0
                    
                    # --- FILTRAGE DU BRUIT (Optionnel) ---
                    if self.denoise_strength > 0:
                        # On utilise un flou Gaussien léger pour lisser le bruit de capteur
                        # kernel_size doit être impair
                        k = (self.denoise_strength * 2) + 1
                        frame = cv2.GaussianBlur(frame, (k, k), 0)
                    
                    # Parallelizing YOLO and Density models
                    t_loop_start = time.time()

                    if not self.yolo_counter:
                        print("[ERROR] Le compteur YOLO n'est pas initialisé, arrêt du pipeline.")
                        break
                    if self.density_enabled and not self.processor:
                        print("[ERROR] Le processeur density n'est pas initialisé, arrêt du pipeline.")
                        break
                    
                    # 1. YOLO INFERENCE (Tiling optionale)
                    preprocess_start = time.monotonic()
                    thr_yolo = ModelThread(
                        self.yolo_counter.count_people, 
                        frame, 
                        tile_size=640, 
                        draw_boxes=True, 
                        use_tiling=self.yolo_tiling,
                        draw_tiles=self.debug_tiling,
                        global_tile_only=self.yolo_global_tile_only,
                    )

                    yolo_label = getattr(self.yolo_counter, 'model_name', 'YOLO')

                    # 2. DENSITY INFERENCE (Tiling 2x2 Batché)
                    t_loop_start = time.time()
                    thr_density = None
                    if self.density_enabled:
                        self.density_quads_coords = []
                        if self.density_tiling:
                            # On prépare les quadrants
                            h, w = frame.shape[:2]
                            mid_h, mid_w = h // 2, w // 2
                            quads = [
                                frame[0:mid_h, 0:mid_w],
                                frame[0:mid_h, mid_w:w],
                                frame[mid_h:h, 0:mid_w],
                                frame[mid_h:h, mid_w:w]
                            ]
                            self.density_quads_coords = [
                                (0, 0, mid_w, mid_h),
                                (mid_w, 0, w, mid_h),
                                (0, mid_h, mid_w, h),
                                (mid_w, mid_h, w, h)
                            ]
                            thr_density = ModelThread(self.processor.process_batch, quads)
                        else:
                            # Mode standard (Inférence globale redimensionnée)
                            thr_density = ModelThread(self.processor.process, frame)

                    preprocess_duration = time.monotonic() - preprocess_start
                    # On lance les deux threads
                    t_cycle_start = time.time()
                    thr_yolo.start()
                    if thr_density:
                        thr_density.start()
                    
                    # Attente YOLO
                    thr_yolo.join()
                    if thr_yolo.result is not None:
                        yolo_count, frame_with_bbox = thr_yolo.result
                    else:
                        print("[ERROR] YOLO thread returned None.")
                        yolo_count, frame_with_bbox = 0, frame.copy()
                    pipeline_report = {}
                    if self.yolo_counter and hasattr(self.yolo_counter, 'get_pipeline_report'):
                        pipeline_report = self.yolo_counter.get_pipeline_report() or {}
                        self.latest_pipeline_report = pipeline_report
                        self.latest_metrics['yolo_pipeline_mode'] = pipeline_report.get('active_mode', 'unknown')
                        if self.extreme_debug and pipeline_report:
                            print(f"[DEBUG] YOLO pipeline report: {pipeline_report}")
                    
                    # Attente Density
                    if thr_density:
                        thr_density.join()

                    postprocess_start = time.monotonic()

                    if self.use_gui:
                        try:
                            win_rect = cv2.getWindowImageRect("People Count")
                            win_w, win_h = (win_rect[2], win_rect[3]) if win_rect and win_rect[2] > 0 else (1280, 720)
                        except Exception:
                            win_w, win_h = 1280, 720
                    else:
                        win_w, win_h = 1280, 720

                    density_result = thr_density.result if thr_density else None
                    density_count = self._get_density_count(density_result)
                    density_heatmap_payload = self._build_density_heatmap_payload(density_result, frame.shape[:2])
                    density_tile_count = len(density_heatmap_payload.get('tiles', [])) if density_heatmap_payload else 0
                    density_ready = bool(density_heatmap_payload and density_tile_count)
                    if self.density_enabled:
                        if density_ready:
                            self._density_payload_empty_logged = False
                        elif not self._density_payload_empty_logged:
                            print("[WARN] Density enabled but heatmap payload produced no tiles; overlay will stay empty until the model returns data.")
                            self._density_payload_empty_logged = True

                    self._overlay_stats = {
                        "compose_ms": None,
                        "preview_ms": None,
                        "cpu_preview_ms": None,
                        "draw_kernel_ms": None,
                        "draw_blend_ms": None,
                        "draw_convert_ms": None,
                        "draw_total_ms": None,
                        "draw_returned_tensor": False,
                        "renderer_fallback_reason": None,
                    }
                    preview_payload = None
                    preview_frame = frame_with_bbox
                    if torch is not None and torch.is_tensor(preview_frame):
                        preview_frame = preview_frame.detach().permute(1, 2, 0).cpu().numpy()

                    yolo_time = thr_yolo.duration
                    density_time = thr_density.duration if thr_density else 0.0

                    yolo_dev = getattr(self.yolo_counter, 'last_device', 'GPU')
                    dens_dev = 'disabled' if not self.density_enabled else getattr(self.processor, 'last_device', 'GPU')
                    yp = getattr(self.yolo_counter, 'last_perf', {}) if self.yolo_counter else {}
                    yolo_internal = getattr(self.yolo_counter, 'last_internal', {}) if self.yolo_counter else {}
                    dp = getattr(self.processor, 'last_perf', {}) if self.processor else {}

                    renderer_draw_kernel = yolo_internal.get('t_draw_kernel_ms')
                    renderer_draw_blend = yolo_internal.get('t_draw_blend_ms')
                    renderer_draw_convert = yolo_internal.get('t_draw_convert_ms')
                    renderer_draw_total = yolo_internal.get('t_draw_total_ms')
                    renderer_returned_tensor = bool(yolo_internal.get('renderer_returned_tensor'))
                    renderer_fallback_reason = yolo_internal.get('renderer_fallback_reason')
                    self._overlay_stats.update({
                        "draw_kernel_ms": renderer_draw_kernel,
                        "draw_blend_ms": renderer_draw_blend,
                        "draw_convert_ms": renderer_draw_convert,
                        "draw_total_ms": renderer_draw_total,
                        "draw_returned_tensor": renderer_returned_tensor,
                        "renderer_fallback_reason": renderer_fallback_reason,
                    })
                    compose_ms = self._overlay_stats.get("compose_ms")
                    threshold_ms = float(os.environ.get('GPU_OVERLAY_WARN_MS', '18.0'))
                    compose_breach = compose_ms is not None and compose_ms > threshold_ms
                    draw_breach = renderer_draw_total is not None and renderer_draw_total > threshold_ms
                    if self.extreme_debug and (compose_breach or draw_breach):
                        compose_label = f"{compose_ms:.1f}" if compose_ms is not None else '—'
                        draw_label = f"{renderer_draw_total:.1f}" if renderer_draw_total is not None else '—'
                        print(f"[GPU PERF] Overlay compose={compose_label}ms draw={draw_label}ms")

                    cpu_usage = psutil.cpu_percent()

                    capture_time = getattr(self.capture, 'last_capture_time', 0.0)
                    if self.extreme_debug:
                        yp_engine = getattr(self.yolo_counter, 'engine', None)
                        yp_debug = getattr(yp_engine, 'last_perf', {}) if yp_engine else yp
                        dp_debug = getattr(self.processor, 'last_perf', {})
                        print(f"[DEBUG] PERF BREAKDOWN (ms):")
                        print(f"  Capture: {capture_time*1000:.1f}ms")
                        print(f"  YOLO {self.yolo_counter.model_name}: Pre={yp_debug.get('preprocess',0)*1000:.1f} | GPU={yp_debug.get('infer',0)*1000:.1f} | Post={yp_debug.get('postprocess',0)*1000:.1f}")
                        print(f"  DENS {self.processor.model_name}: Pre={dp_debug.get('preprocess',0)*1000:.1f} | GPU={dp_debug.get('infer',0)*1000:.1f} | Post={dp_debug.get('postprocess',0)*1000:.1f}")

                    total_process_time = time.time() - t_loop_start

                    if preview_frame is None:
                        preview_frame = frame_with_bbox.detach().permute(1, 2, 0).cpu().numpy() if torch is not None and torch.is_tensor(frame_with_bbox) else frame_with_bbox

                    if self.frame_callback:
                        if preview_payload is not None:
                            self.frame_callback(preview_payload.to_payload())
                        else:
                            self.frame_callback(preview_frame)

                    total_time = time.time() - start_time
                    fps = 1.0 / total_time if total_time > 0 else 0
                    avg_instant = (yolo_count + density_count) / 2.0
                    self.metrics_collector.append(yolo_count, density_count, avg_instant)
                    history_snapshot = self.metrics_collector.snapshot() if self.graph_overlay_enabled else []
                    detection_payload = None
                    if self.yolo_counter and hasattr(self.yolo_counter, 'get_detection_payload'):
                        detection_payload = self.yolo_counter.get_detection_payload()
                    yolo_profiler_ms = getattr(self.yolo_counter, 'last_cuda_profiler_ms', None)
                    density_profiler_ms = getattr(self.processor, 'last_cuda_profiler_ms', None)
                    metrics = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "yolo_count": int(yolo_count),
                        "density_count": float(density_count),
                        "density_enabled": bool(self.density_enabled),
                        "average": float(avg_instant),
                        "fps": float(fps),
                        "cpu_usage": float(cpu_usage),
                        "total_time": float(total_process_time),
                        "yolo_time": float(yolo_time),
                        "density_time": float(density_time),
                        "density_ready": bool(density_ready),
                        "density_tile_count": int(density_tile_count),
                        "yolo_device": self.yolo_device,
                        "density_device": self.density_device,
                        "debug_mode": bool(self.extreme_debug),
                        "profile_active": bool(self.profile_active),
                        "profile_name": self.active_profile_name or 'Live Metrics',
                        "graph_overlay": bool(self.graph_overlay_enabled),
                        "capture_ms": float(capture_time * 1000),
                        "pipeline_pre_ms": float(preprocess_duration * 1000),
                        "pipeline_post_ms": float(postprocess_duration * 1000),
                        "yolo_pre_ms": float(yp.get('preprocess', 0) * 1000),
                        "yolo_gpu_ms": float(yp.get('infer', 0) * 1000),
                        "yolo_post_ms": float(yp.get('postprocess', 0) * 1000),
                        "yolo_cuda_profiler_ms": float(yolo_profiler_ms) if yolo_profiler_ms is not None else None,
                        "yolo_pipeline_mode": self.yolo_pipeline_mode,
                        "yolo_pipeline_mode_effective": getattr(self.yolo_counter, 'active_pipeline_mode', self.yolo_pipeline_mode),
                        "yolo_preproc_mode": getattr(self.yolo_counter, 'preprocessor_mode', 'cpu'),
                        "yolo_renderer_mode": getattr(self.yolo_counter, 'renderer_mode', 'cpu'),
                        "yolo_internal_inf_ms": float(yolo_internal.get('t_inf', 0) * 1000),
                        "yolo_internal_draw_ms": float(yolo_internal.get('t_draw', 0) * 1000),
                        "yolo_internal_crop_ms": float(yolo_internal.get('t_crop', 0) * 1000),
                        "yolo_internal_fuse_ms": float(yolo_internal.get('t_fuse', 0) * 1000),
                        "yolo_internal_total_ms": float(yolo_internal.get('total_internal', 0) * 1000),
                        "overlay_compose_ms": self._overlay_stats.get("compose_ms"),
                        "overlay_draw_kernel_ms": self._overlay_stats.get("draw_kernel_ms"),
                        "overlay_draw_blend_ms": self._overlay_stats.get("draw_blend_ms"),
                        "overlay_draw_convert_ms": self._overlay_stats.get("draw_convert_ms"),
                        "overlay_draw_total_ms": self._overlay_stats.get("draw_total_ms"),
                        "overlay_draw_returned_tensor": self._overlay_stats.get("draw_returned_tensor"),
                        "overlay_renderer_fallback": self._overlay_stats.get("renderer_fallback_reason"),
                        "overlay_preview_ms": self._overlay_stats.get("preview_ms"),
                        "overlay_cpu_preview_ms": self._overlay_stats.get("cpu_preview_ms"),
                        "density_pre_ms": float(dp.get('preprocess', 0) * 1000),
                        "density_gpu_ms": float(dp.get('infer', 0) * 1000),
                        "density_post_ms": float(dp.get('postprocess', 0) * 1000),
                        "density_cuda_profiler_ms": float(density_profiler_ms) if density_profiler_ms is not None else None,
                        "history": history_snapshot,
                        "yolo_pipeline_report": pipeline_report
                    }
                    if detection_payload is not None:
                        metrics["yolo_detection_payload"] = detection_payload
                    else:
                        metrics["yolo_detection_payload"] = {"clusters": [], "detections": []}
                    mask_payload = None
                    mask_payload_created_ts = None
                    mask_payload_created_iso = None
                    if self.yolo_counter and hasattr(self.yolo_counter, 'get_mask_payload'):
                        mask_payload = self.yolo_counter.get_mask_payload()
                        if mask_payload is not None:
                            mask_payload_created_ts = mask_payload.get('created_at_ts')
                            mask_payload_created_iso = mask_payload.get('created_at')
                    metrics["yolo_mask_payload"] = mask_payload
                    metrics["yolo_mask_payload_created_at"] = mask_payload_created_iso
                    metrics["yolo_mask_payload_created_ts"] = mask_payload_created_ts
                    mask_sent_ts = time.time()
                    mask_sent_iso = datetime.utcnow().isoformat() + "Z"
                    metrics["yolo_mask_payload_sent_at"] = mask_sent_iso
                    if mask_payload_created_ts is not None:
                        metrics["yolo_mask_payload_latency_ms"] = (mask_sent_ts - mask_payload_created_ts) * 1000.0
                    else:
                        metrics["yolo_mask_payload_latency_ms"] = None
                    metrics["density_heatmap_payload"] = density_heatmap_payload
                    heatmap_payload_created_iso = density_heatmap_payload.get('created_at') if density_heatmap_payload else None
                    heatmap_payload_created_ts = density_heatmap_payload.get('created_at_ts') if density_heatmap_payload else None
                    heatmap_payload_sent_ts = time.time()
                    heatmap_payload_sent_iso = datetime.utcnow().isoformat() + "Z"
                    metrics["density_heatmap_payload_created_at"] = heatmap_payload_created_iso
                    metrics["density_heatmap_payload_created_ts"] = heatmap_payload_created_ts
                    metrics["density_heatmap_payload_sent_at"] = heatmap_payload_sent_iso
                    if heatmap_payload_created_ts is not None:
                        metrics["density_heatmap_payload_latency_ms"] = (heatmap_payload_sent_ts - heatmap_payload_created_ts) * 1000.0
                    else:
                        metrics["density_heatmap_payload_latency_ms"] = None
                    if mask_payload_created_ts is not None:
                        latency_ms = metrics["yolo_mask_payload_latency_ms"]
                        print(f"[MASK TIMING] created={mask_payload_created_iso} sent={mask_sent_iso} latency={latency_ms:.1f}ms")
                    self.latest_metrics = metrics
                    if self.profile_active:
                        self.profile_log.append(metrics.copy())
                        if len(self.profile_log) > 300:
                            self.profile_log.pop(0)
                    if self.metrics_callback:
                        try:
                            self.metrics_callback(metrics)
                        except Exception as exc:
                            print(f"[WARN] Metrics callback failed: {exc}")
                    
                    if self.extreme_debug:
                        summary = {
                            "fps": round(fps, 2),
                            "cpu_usage": round(cpu_usage, 1),
                            "capture_ms": round(capture_time * 1000, 1),
                            "pipeline_pre_ms": round(preprocess_duration * 1000, 1),
                            "pipeline_post_ms": round(postprocess_duration * 1000, 1),
                            "yolo_time_ms": round(yolo_time * 1000, 1),
                            "density_time_ms": round(density_time * 1000, 1),
                            "mask_latency_ms": metrics.get("yolo_mask_payload_latency_ms"),
                            "mask_created_at": metrics.get("yolo_mask_payload_created_at"),
                            "mask_sent_at": metrics.get("yolo_mask_payload_sent_at"),
                            "density_heatmap_payload_latency_ms": metrics.get("density_heatmap_payload_latency_ms"),
                            "density_heatmap_payload_created_at": metrics.get("density_heatmap_payload_created_at"),
                            "density_heatmap_payload_sent_at": metrics.get("density_heatmap_payload_sent_at"),
                            "density_ready": metrics.get("density_ready"),
                            "density_tile_count": metrics.get("density_tile_count"),
                            "preprocessor_mode": metrics.get("yolo_preproc_mode"),
                            "renderer_mode": metrics.get("yolo_renderer_mode"),
                            "yolo_pipeline_mode": metrics.get("yolo_pipeline_mode"),
                            "yolo_cuda_profiler_ms": metrics.get("yolo_cuda_profiler_ms"),
                            "density_cuda_profiler_ms": metrics.get("density_cuda_profiler_ms"),
                        }
                        print(f"[DEBUG] AGGREGATED METRICS {json.dumps(summary)}")

                    if self.use_gui:
                        cv2.imshow("People Count", preview_frame)

                    postprocess_duration = time.monotonic() - postprocess_start
                    # Update mini-graph history at every frame for live feedback (YOLO, Density, Average)
                    self.history_data.append((yolo_count, density_count, avg_instant))
                    if len(self.history_data) > self.history_len:
                        self.history_data.pop(0)

                    counts_yolo.append(yolo_count)
                    counts_density.append(density_count)
                    now_sec = int(time.time())
                    if now_sec - last_period >= AVERAGING_PERIOD_SECONDS:
                        if counts_yolo and counts_density:
                            avg_yolo = sum(counts_yolo) / len(counts_yolo)
                            avg_density = sum(counts_density) / len(counts_density)
                            avg_combined = (avg_yolo + avg_density) / 2.0
                            max_count = max(avg_yolo, avg_density)
                            now_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            if self.extreme_debug:
                                print(f"Average YOLO ({yolo_label}) People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_yolo:.2f}")
                                print(f"Average Density People (last {AVERAGING_PERIOD_SECONDS} sec): {avg_density:.2f}")
                                print(f"Average Combined (last {AVERAGING_PERIOD_SECONDS} sec): {avg_combined:.2f}")
                                print(f"Max People Count (last {AVERAGING_PERIOD_SECONDS} sec): {max_count:.2f}")
                            writer.writerow([now_dt, f"{avg_yolo:.2f}", f"{avg_density:.2f}", f"{avg_combined:.2f}", f"{max_count:.2f}"])
                            csvfile.flush()
                            log_data.append((now_dt, avg_yolo, avg_density, avg_combined, max_count))
                        
                        counts_yolo = []
                        counts_density = []
                        last_period = now_sec
                    
                    if self.use_gui:
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q') or cv2.getWindowProperty("People Count", cv2.WND_PROP_VISIBLE) < 1:
                            break
                    else:
                        # En mode headless, on ajoute un petit sleep pour ne pas saturer le CPU si FPS > 100
                        time.sleep(0.01)
            finally:
                self.capture.release()
                if self.use_gui:
                    cv2.destroyAllWindows()
                print("Camera released and application closed.")
                # Génère la courbe à partir du CSV
                if log_data:
                    times = [dt for dt, _, _, _, _ in log_data]
                    yolo_vals = [y for _, y, _, _, _ in log_data]
                    density_vals = [d for _, _, d, _, _ in log_data]
                    avg_combined_vals = [a for _, _, _, a, _ in log_data]
                    
                    plt.figure(figsize=(14, 7))
                    plt.plot(times, yolo_vals, marker='o', markersize=4, label='YOLO', color='green', alpha=0.6)
                    plt.plot(times, density_vals, marker='x', markersize=4, label='Density', color='red', alpha=0.6)
                    plt.plot(times, avg_combined_vals, marker='d', markersize=4, label='Average (Combined)', color='cyan', linewidth=2)
                    
                    # Affiche un label de temps toutes les N points pour éviter la surcharge
                    N = max(1, len(times)//20)
                    plt.xticks([i for i in range(0, len(times), N)], [times[i] for i in range(0, len(times), N)], rotation=45)
                    plt.xlabel('Time')
                    plt.ylabel('Average People Count')
                    plt.title('People Count Over Time')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(img_path)
                    print(f"Saved plot to {img_path}")

if __name__ == "__main__":
    sys.path.append(r"E:\AI\lwcc")
    # Allow override by environment variable CAPTURE_MODE or by CLI first arg
    capture_mode = None
    if len(sys.argv) > 1:
        capture_mode = sys.argv[1]
    app = CameraAppPipeline(capture_mode=capture_mode)
    app.run()
