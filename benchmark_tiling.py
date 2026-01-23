import time
import os
import sys
import numpy as np
import cv2
import psutil
# NVML disabled to avoid driver conflicts during benchmark
HAS_NVML = False

# Import des composants de PeopleCounter
from camera_capture import CameraCapture
from yolo_people_counter import YoloPeopleCounter
from people_counter_processor import PeopleCounterProcessor

def get_gpu_util():
    if HAS_NVML:
        try:
            util = nvmlDeviceGetUtilizationRates(nvml_handle)
            return util.gpu
        except: return 0
    return 0

def benchmark_run(resolution, model_name, yolo_tiling, density_tiling, duration=15):
    print(f"\n{'='*60}")
    print(f"RUN: Res={resolution}, YOLO_Tile={yolo_tiling}, Dens_Tile={density_tiling}")
    print(f"{'='*60}")
    
    # Init Models
    yolo = YoloPeopleCounter(model_name=model_name, device='cuda', backend='tensorrt_native')
    density = PeopleCounterProcessor(model_name="CSRNet", model_weights="SHA", backend="tensorrt")
    
    # Init Capture
    w, h = (3840, 2160) if resolution == "4k" else (1920, 1080)
    cap = CameraCapture(width=w, height=h)
    
    metrics = []
    start_bench = time.time()
    
    while (time.time() - start_bench) < duration:
        frame = cap.get_frame()
        if frame is None: continue
        
        t0 = time.time()
        
        # 1. YOLO
        t_yolo_s = time.time()
        y_res = yolo.count_people(frame, use_tiling=yolo_tiling)
        y_time = time.time() - t_yolo_s
        
        # 2. Density
        t_dens_s = time.time()
        if density_tiling:
            h, w = frame.shape[:2]
            mid_h, mid_w = h // 2, w // 2
            quads = [frame[0:mid_h, 0:mid_w], frame[0:mid_h, mid_w:w], frame[mid_h:h, 0:mid_w], frame[mid_h:h, mid_w:w]]
            _ = density.process_batch(quads)
        else:
            _ = density.process(frame)
        d_time = time.time() - t_dens_s
        
        total_t = time.time() - t0
        
        metrics.append({
            'total': total_t,
            'yolo': y_time,
            'density': d_time,
            'cpu': psutil.cpu_percent(),
            'gpu': get_gpu_util()
        })
    
    cap.release()
    
    # Analyse des 10 derniers échantillons
    last_samples = metrics[-10:] if len(metrics) > 10 else metrics
    if not last_samples: return None
    
    avg_total = sum(m['total'] for m in last_samples) / len(last_samples)
    avg_yolo = sum(m['yolo'] for m in last_samples) / len(last_samples)
    avg_dens = sum(m['density'] for m in last_samples) / len(last_samples)
    avg_cpu = sum(m['cpu'] for m in last_samples) / len(last_samples)
    avg_gpu = sum(m['gpu'] for m in last_samples) / len(last_samples)
    
    return {
        'fps': 1.0 / avg_total,
        'yolo_s': avg_yolo,
        'dens_s': avg_dens,
        'total_s': avg_total,
        'cpu': avg_cpu,
        'gpu': avg_gpu
    }

if __name__ == "__main__":
    results = []
    target_model = "yolo12s.engine" # Plus rapide pour bench
    
    # Init Models ONCE outside the loop to avoid resource exhaustion
    print("Initializing models...")
    yolo = YoloPeopleCounter(model_name=target_model, device='cuda', backend='tensorrt_native')
    density = PeopleCounterProcessor(model_name="CSRNet", model_weights="SHA", backend="tensorrt")
    
    configs = [
        # (Res, Y_Tile, D_Tile)
        ("1080p", False, False),
        ("1080p", True, False),
        ("1080p", False, True),
        ("1080p", True, True),
        ("4k", False, False),
        ("4k", True, False),
        ("4k", False, True),
        ("4k", True, True),
    ]
    
    for res, yt, dt in configs:
        # Pass pre-initialized models
        def benchmark_run_optimized(resolution, yolo, density, yolo_tiling, density_tiling, duration=10):
            print(f"\n{'='*60}")
            print(f"RUN: Res={resolution}, YOLO_Tile={yolo_tiling}, Dens_Tile={density_tiling}")
            print(f"{'='*60}")
            
            # Init Capture
            w, h = (3840, 2160) if resolution == "4k" else (1920, 1080)
            cap = CameraCapture(width=w, height=h)
            
            metrics = []
            start_bench = time.time()
            
            # Flush existing frames
            for _ in range(5): cap.get_frame()
            
            while (time.time() - start_bench) < duration:
                frame = cap.get_frame()
                if frame is None:
                    print("Camera returned None frame")
                    continue
                
                t0 = time.time()
                
                # 1. YOLO
                t_yolo_s = time.time()
                print("DEBUG: Starting YOLO count_people...")
                _ = yolo.count_people(frame, use_tiling=yolo_tiling)
                print("DEBUG: YOLO done.")
                y_time = time.time() - t_yolo_s
                
                # 2. Density
                t_dens_s = time.time()
                print("DEBUG: Starting Density process...")
                if density_tiling:
                    h_f, w_f = frame.shape[:2]
                    mid_h, mid_w = h_f // 2, w_f // 2
                    quads = [frame[0:mid_h, 0:mid_w], frame[0:mid_h, mid_w:w_f], frame[mid_h:h_f, 0:mid_w], frame[mid_h:h_f, mid_w:w_f]]
                    _ = density.process_batch(quads)
                else:
                    _ = density.process(frame)
                print("DEBUG: Density done.")
                d_time = time.time() - t_dens_s
                
                total_t = time.time() - t0
                # print(f"DEBUG: Frame done in {total_t:.3f}s")
                
                metrics.append({
                    'total': total_t,
                    'yolo': y_time,
                    'density': d_time,
                    'cpu': psutil.cpu_percent(),
                    'gpu': get_gpu_util()
                })
            
            cap.release()
            
            # Analyse des 10 derniers échantillons
            last_samples = metrics[-10:] if len(metrics) > 10 else metrics
            if not last_samples: return None
            
            avg_total = sum(m['total'] for m in last_samples) / len(last_samples)
            avg_yolo = sum(m['yolo'] for m in last_samples) / len(last_samples)
            avg_dens = sum(m['density'] for m in last_samples) / len(last_samples)
            avg_cpu = sum(m['cpu'] for m in last_samples) / len(last_samples)
            avg_gpu = sum(m['gpu'] for m in last_samples) / len(last_samples)
            
            return {
                'fps': 1.0 / avg_total,
                'yolo_s': avg_yolo,
                'dens_s': avg_dens,
                'total_s': avg_total,
                'cpu': avg_cpu,
                'gpu': avg_gpu
            }

        res_data = benchmark_run_optimized(res, yolo, density, yt, dt)
        if res_data:
            res_data.update({'res': res, 'yt': yt, 'dt': dt})
            results.append(res_data)
            print(f"RESULT: FPS={res_data['fps']:.2f} | CPU={res_data['cpu']}% | GPU={res_data['gpu']}%")
        time.sleep(2) # Cooldown between runs

    print("\n\n" + "#"*40)
    print("      FINAL BENCHMARK SUMMARY")
    print("#"*40)
    print(f"{'Mode':<25} | {'FPS':<6} | {'YOLO(s)':<8} | {'DENS(s)':<8} | {'CPU%':<5} | {'GPU%':<5}")
    print("-" * 75)
    for r in results:
        mode_str = f"{r['res']} Y:{int(r['yt'])} D:{int(r['dt'])}"
        print(f"{mode_str:<25} | {r['fps']:<6.2f} | {r['yolo_s']:<8.3f} | {r['dens_s']:<8.3f} | {r['cpu']:<5.1f} | {r['gpu']:<5.1f}")
