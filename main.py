from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Query, Response
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="avart-engine",
    version="0.2.0",
    description="Poster engine (stroke preview). Upload a profile photo on white background → outer contour line.",
)

# ✅ CORS: tillad din one.com-side at kalde API’et fra browseren
# (tilpas når du kender dit endelige domæne)
ALLOWED_ORIGINS = [
    "https://avart.dk",
    "https://www.avart.dk",
    "https://avart-poster.dk",
    "https://www.avart-poster.dk",
    "http://localhost:3000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to BGR image."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    return img


def _outer_contour_mask_for_white_bg(
    bgr: np.ndarray,
    *,
    white_threshold: int = 235,
    min_area_ratio: float = 0.02,
) -> np.ndarray:
    """
    Create a binary mask for the main subject assuming a *light/white background*.
    Returns mask with subject = 255, background = 0.
    """
    h, w = bgr.shape[:2]

    # 1) Convert to grayscale (stabilt for hvide baggrunde)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2) Find background-ish pixels (nær hvid)
    # Background = 1 where gray >= white_threshold
    bg = (gray >= white_threshold).astype(np.uint8) * 255

    # 3) Invert to get rough foreground
    fg = cv2.bitwise_not(bg)

    # 4) Cleanup / fill holes (morfologi)
    k = max(3, int(min(h, w) * 0.01) | 1)  # odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5) Find largest contour (typisk person)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fg  # fallback

    areas = [cv2.contourArea(c) for c in contours]
    largest_idx = int(np.argmax(areas))
    largest = contours[largest_idx]

    # 6) Reject if too small → return raw fg
    if cv2.contourArea(largest) < (h * w * min_area_ratio):
        return fg

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def _render_outline_png(
    mask: np.ndarray,
    *,
    thickness_px: int = 6,
    smooth: bool = True,
) -> bytes:
    """
    Render only the OUTER contour as black stroke on white background.
    Returns PNG bytes.
    """
    h, w = mask.shape[:2]

    # optional smoothing to make contour less "jagged"
    m = mask.copy()
    if smooth:
        # blur + rethreshold
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0, sigmaY=2.0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background
    if contours:
        # Draw ONLY outer contours
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), thickness=thickness_px, lineType=cv2.LINE_AA)

    ok, png = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Failed to encode PNG.")
    return png.tobytes()


@app.post("/stroke/preview", summary="Stroke Preview")
async def stroke_preview(
    file: UploadFile = File(...),
    thickness: int = Query(6, ge=1, le=40, description="Stroke thickness in pixels (preview)"),
    white_threshold: int = Query(235, ge=200, le=255, description="How 'white' the background must be to be treated as background"),
    smooth: bool = Query(True, description="Smoother contour lines"),
):
    """
    Upload a profile photo (ideally on white background).
    Returns a PNG with only the outer silhouette contour (stroke).
    """
    data = await file.read()
    bgr = _read_image(data)

    # Build mask and outline
    mask = _outer_contour_mask_for_white_bg(bgr, white_threshold=white_threshold)
    png_bytes = _render_outline_png(mask, thickness_px=thickness, smooth=smooth)

    return Response(content=png_bytes, media_type="image/png")
