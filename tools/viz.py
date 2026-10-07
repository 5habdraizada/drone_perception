"""Drawing helpers shared by the demos and tools."""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = np.array([(80, 80, 80), (60, 180, 75), (230, 25, 75), (255, 225, 25), (0, 130, 200),
                    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
                    (250, 190, 212), (0, 128, 128)], np.uint8)[:, ::-1]  # BGR


def colourise(values: np.ndarray, lo: float, hi: float, cmap=cv2.COLORMAP_TURBO,
              size: tuple[int, int] | None = None) -> np.ndarray:
    """Float map -> BGR image; NaN shown dark grey."""
    norm = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    img[~np.isfinite(values)] = 40
    return img if size is None else cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


def regions_bgr(regions: np.ndarray) -> np.ndarray:
    return PALETTE[np.maximum(regions, 0) % len(PALETTE)].copy()


def overlay(gray: np.ndarray, cells_bgr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a per-cell colour map over a grayscale image of any size."""
    big = cv2.resize(cells_bgr, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1 - alpha, big, alpha, 0)


def label(img: np.ndarray, text: str, size: tuple[int, int] = (320, 176)) -> np.ndarray:
    out = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), size)
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out
