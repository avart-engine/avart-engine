from __future__ import annotations

import io
from typing import List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

app = FastAPI(
    title="avart-engine",
    version="0.3.0",
    description="STEP 1: single-image silhouette preview + SVG",
)

# Midlertidigt åben CORS mens vi bygger
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


# ----------------------------
# Helpers
# ----------------------------

def read_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload JPG or PNG.")
    return img


def estimate_background_from_corners(bgr: np.ndarray) -> np.ndarray:
    """
    Estimate background color from image corners.
    Works reasonably well for light / even backgrounds.
    Returns BGR median color.
    """
    h, w = bgr.shape[:2]
    s = max(10, min(h, w) // 12)

    patches = [
        bgr[0:s, 0:s],
        bgr[0:s, w - s:w],
        bgr[h - s:h, 0:s],
        bgr[h - s:h, w - s:w],
    ]

    samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    bg = np.median(samples, axis=0)
    return bg.astype(np.uint8)


def create_subject_mask(
    bgr: np.ndarray,
    white_threshold: int = 235,
    corner_sample: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Create binary mask:
    subject = 255
    background = 0
    """
    h, w = bgr.shape[:2]

    if corner_sample:
        bg = estimate_background_from_corners(bgr).astype(np.int16)
        diff = bgr.astype(np.int16) - bg
        dist = np.sqrt(np.sum(diff * diff, axis=2))

        # Map "white_threshold" to a practical distance threshold.
        # Higher white_threshold => stricter background detection.
        dist_threshold = max(8, 255 - white_threshold)
        mask = (dist > dist_threshold).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray < white_threshold).astype(np.uint8) * 255

    if smooth:
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clean = np.zeros((h, w), dtype=np.uint8)

    if not contours:
        return clean

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)

    return clean


def smooth_contour(contour: np.ndarray, window: int = 17) -> np.ndarray:
    """
    Moving-average smoothing over contour points.
    Keeps closed shape and removes some jaggedness.
    """
    pts = contour[:, 0, :].astype(np.float32)
    n = len(pts)

    if n < window or n < 20:
        return contour

    if window % 2 == 0:
        window += 1

    pad = window // 2
    pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])

    smoothed = []
    for i in range(n):
        seg = pts_pad[i:i + window]
        smoothed.append(seg.mean(axis=0))

    smoothed = np.array(smoothed, dtype=np.int32).reshape(-1, 1, 2)
    return smoothed


def get_smoothed_outer_contour(
    mask: np.ndarray,
    epsilon_ratio: float = 0.0015,
    smooth_window: int = 17,
) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found")

    largest = max(contours, key=cv2.contourArea)

    # Simplify slightly
    peri = cv2.arcLength(largest, True)
    eps = max(1.0, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(largest, eps, True)

    # Smooth
    smooth = smooth_contour(approx, window=smooth_window)
    return smooth


def render_preview_png(
    contour: np.ndarray,
    width: int,
    height: int,
    thickness: int = 2,
    upscale: int = 4,
) -> bytes:
    """
    Draw contour on white background.
    Render large first, then scale down for smoother line.
    """
    W = width * upscale
    H = height * upscale

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    c = contour.copy().astype(np.int32)
    c[:, 0, 0] *= upscale
    c[:, 0, 1] *= upscale

    cv2.drawContours(
        canvas,
        [c],
        -1,
        (0, 0, 0),
        thickness=max(1, thickness * upscale),
        lineType=cv2.LINE_AA,
    )

    canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Could not encode PNG")
    return png.tobytes()


def contour_to_svg(
    contour: np.ndarray,
    width: int,
    height: int,
    stroke_width: float = 2.0,
) -> str:
    """
    Convert contour to a very simple SVG polygon path.
    This is STEP 1: not perfect bezier yet, but already vector.
    """
    pts = contour[:, 0, :]
    if len(pts) < 3:
        raise ValueError("Contour too small for SVG")

    d = f"M {pts[0,0]} {pts[0,1]} "
    for p in pts[1:]:
        d += f"L {p[0]} {p[1]} "
    d += "Z"

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <path d="{d}" fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>
</svg>'''
    return svg


# ----------------------------
# API
# ----------------------------

@app.post("/silhouette/preview")
async def silhouette_preview(
    file: UploadFile = File(...),
    white_threshold: int = Query(235, ge=150, le=255),
    corner_sample: bool = Query(True),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    smooth_window: int = Query(17, ge=5, le=51),
    thickness: int = Query(2, ge=1, le=8),
    upscale: int = Query(4, ge=1, le=8),
):
    try:
        bgr = read_upload_to_bgr(file)
        h, w = bgr.shape[:2]

        mask = create_subject_mask(
            bgr,
            white_threshold=white_threshold,
            corner_sample=corner_sample,
            smooth=smooth,
        )

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        png = render_preview_png(
            contour,
            width=w,
            height=h,
            thickness=thickness,
            upscale=upscale,
        )

        return Response(content=png, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/silhouette/svg")
async def silhouette_svg(
    file: UploadFile = File(...),
    white_threshold: int = Query(235, ge=150, le=255),
    corner_sample: bool = Query(True),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    smooth_window: int = Query(17, ge=5, le=51),
    stroke_width: float = Query(2.0, ge=0.5, le=10.0),
):
    try:
        bgr = read_upload_to_bgr(file)
        h, w = bgr.shape[:2]

        mask = create_subject_mask(
            bgr,
            white_threshold=white_threshold,
            corner_sample=corner_sample,
            smooth=smooth,
        )

        contour = get_smoothed_outer_contour(
            mask,
            epsilon_ratio=epsilon_ratio,
            smooth_window=smooth_window,
        )

        svg = contour_to_svg(
            contour,
            width=w,
            height=h,
            stroke_width=stroke_width,
        )

        return Response(content=svg, media_type="image/svg+xml")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
