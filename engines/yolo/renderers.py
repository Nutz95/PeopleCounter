from abc import ABC, abstractmethod
from typing import Sequence, Union

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


class BaseMaskRenderer(ABC):
    @abstractmethod
    def render(
        self,
        image: np.ndarray,
        clusters: Sequence[Sequence[int]],
        all_boxes: np.ndarray,
        all_masks: Sequence[np.ndarray],
        all_visible_boxes: Sequence[Sequence[float]],
        metadata: Sequence[tuple],
        draw_boxes: bool,
        debug_mode: bool,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, 'torch.Tensor']:
        raise NotImplementedError()


class CpuMaskRenderer(BaseMaskRenderer):
    def render(
        self,
        image,
        clusters,
        all_boxes,
        all_masks,
        all_visible_boxes,
        metadata,
        draw_boxes,
        debug_mode,
        return_tensor: bool = False,
    ):
        if not draw_boxes or image is None:
            return image

        image_out = image.copy()
        h_img, w_img = image_out.shape[:2]
        combined_mask_full = np.zeros((h_img, w_img), dtype=np.uint8)

        for cluster in clusters:
            cvx1, cvy1, cvx2, cvy2 = 999999, 999999, -999999, -999999
            has_any_mask = False
            person_mask = None

            for idx in cluster:
                bx1, by1, bx2, by2 = all_boxes[idx]
                cvx1, cvy1 = min(cvx1, bx1), min(cvy1, by1)
                cvx2, cvy2 = max(cvx2, bx2), max(cvy2, by2)

            cvx1, cvy1 = max(0, int(cvx1)), max(0, int(cvy1))
            cvx2, cvy2 = min(w_img, int(cvx2)), min(h_img, int(cvy2))
            cw, ch = cvx2 - cvx1, cvy2 - cvy1

            if cw <= 0 or ch <= 0:
                continue

            person_mask = np.zeros((ch, cw), dtype=np.float32)
            debug_rects = []

            for idx in cluster:
                mask_fragment = all_masks[idx]
                if mask_fragment is None:
                    continue
                if torch is not None and torch.is_tensor(mask_fragment):
                    mask_fragment = mask_fragment.cpu().numpy()
                if mask_fragment is None or (hasattr(mask_fragment, 'size') and mask_fragment.size == 0):
                    continue
                has_any_mask = True
                vx1, vy1, vx2, vy2 = [int(v) for v in all_visible_boxes[idx]]
                fx1, fy1 = max(cvx1, vx1), max(cvy1, vy1)
                fx2, fy2 = min(cvx2, vx2), min(cvy2, vy2)
                fw, fh = fx2 - fx1, fy2 - fy1
                if fw <= 0 or fh <= 0:
                    continue
                try:
                    m_resized = cv2.resize(mask_fragment, (fw, fh), interpolation=cv2.INTER_LINEAR)
                    px, py = fx1 - cvx1, fy1 - cvy1
                    target_roi = person_mask[py:py+fh, px:px+fw]
                    if target_roi.shape == m_resized.shape:
                        np.maximum(target_roi, m_resized, out=target_roi)
                except Exception:
                    pass
                debug_rects.append((vx1, vy1, vx2, vy2))

            mask_bin = (person_mask > 0.5).astype(np.uint8)
            if mask_bin.any():
                roi_target = combined_mask_full[cvy1:cvy2, cvx1:cvx2]
                if roi_target.shape == mask_bin.shape:
                    np.maximum(roi_target, mask_bin, out=roi_target)
                    ys, xs = np.where(mask_bin > 0)
                    if ys.size and xs.size:
                        miny, maxy = max(0, int(ys.min()) - 2), min(ch, int(ys.max()) + 2)
                        minx, maxx = max(0, int(xs.min()) - 2), min(cw, int(xs.max()) + 2)
                        mask_roi = mask_bin[miny:maxy, minx:maxx]
                        if mask_roi.any():
                            global_roi = combined_mask_full[cvy1+miny:cvy1+maxy, cvx1+minx:cvx1+maxx]
                            # Blend overlay
                            overlay_slice = image_out[cvy1+miny:cvy1+maxy, cvx1+minx:cvx1+maxx].copy()
                            overlay_slice[mask_roi > 0] = (0, 255, 0)
                            blended_roi = cv2.addWeighted(overlay_slice, 0.4, image_out[cvy1+miny:cvy1+maxy, cvx1+minx:cvx1+maxx], 0.6, 0)
                            mask_bool = mask_roi > 0
                            image_out[cvy1+miny:cvy1+maxy, cvx1+minx:cvx1+maxx][mask_bool] = blended_roi[mask_bool]
                            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                cv2.drawContours(image_out[cvy1+miny:cvy1+maxy, cvx1+minx:cvx1+maxx], contours, -1, (0, 100, 0), 2)

            if debug_mode:
                for tx1, ty1, tx2, ty2 in debug_rects:
                    cv2.rectangle(image_out, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)

        return image_out


class GpuMaskRenderer(BaseMaskRenderer):
    def __init__(self):
        if torch is None or not torch.cuda.is_available():
            raise EnvironmentError("CUDA is required for GPU mask rendering.")
        self.device = torch.device('cuda')

    def render(
        self,
        image,
        clusters,
        all_boxes,
        all_masks,
        all_visible_boxes,
        metadata,
        draw_boxes,
        debug_mode,
        return_tensor: bool = False,
    ):
        if not draw_boxes or image is None:
            return image

        h_img, w_img = image.shape[:2]
        mask_tensor = torch.zeros((h_img, w_img), dtype=torch.bool, device=self.device)
        for idx, mask_fragment in enumerate(all_masks):
            if mask_fragment is None:
                continue
            if torch.is_tensor(mask_fragment):
                if mask_fragment.numel() == 0:
                    continue
            else:
                if getattr(mask_fragment, 'size', 0) == 0:
                    continue
            vx1, vy1, vx2, vy2 = [int(v) for v in all_visible_boxes[idx]]
            h_roi = vy2 - vy1
            w_roi = vx2 - vx1
            if h_roi <= 0 or w_roi <= 0:
                continue
            if torch.is_tensor(mask_fragment):
                mask_tensor_roi = mask_fragment.to(dtype=torch.float32, device=self.device)
            else:
                mask_tensor_roi = torch.from_numpy(mask_fragment.astype(np.float32)).to(self.device)
            mask_tensor_roi = mask_tensor_roi.unsqueeze(0).unsqueeze(0)
            if F is not None and mask_tensor_roi.shape[-2:] != (h_roi, w_roi):
                mask_tensor_roi = F.interpolate(mask_tensor_roi, size=(h_roi, w_roi), mode='bilinear', align_corners=False)
            else:
                mask_tensor_roi = mask_tensor_roi[:, :, :h_roi, :w_roi]
            mask_bool = mask_tensor_roi.squeeze(0).squeeze(0) > 0.5
            if mask_bool.numel() == 0:
                continue
            mask_tensor[vy1:vy2, vx1:vx2] |= mask_bool

        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).to(self.device)
        overlay_color = torch.tensor([0, 255, 0], dtype=torch.float32, device=self.device).view(3, 1, 1)
        blend = image_tensor * 0.6 + overlay_color * 0.4
        mask_expanded = mask_tensor.unsqueeze(0).expand(3, -1, -1)
        image_tensor = torch.where(mask_expanded, blend, image_tensor)
        rendered = image_tensor.clamp(0, 255).byte()
        if return_tensor:
            return rendered
        return rendered.permute(1, 2, 0).cpu().numpy()
