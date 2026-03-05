from __future__ import annotations

import io
from typing import Literal, Tuple

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = FastAPI(title="avart-engine", version="0.2.0")


# -------------------------
# Helpers
# -------------------------

def _read_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    """Read UploadFile into OpenCV BGR image."""
    data = upload.file.read()
    if not data:
        raise ValueError("Empty upload.")
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload JPG/PNG.")
    return img


def _auto_white_mask_from_background(
    bgr: np.ndarray,
    white_threshold: int = 235,
    corner_sample: bool = True,
) -> np.ndarray:
    """
    Create a foreground mask assuming a light background.
    If corner_sample=True, we estimate background color from corners and threshold by distance.
    Else we use a simple 'near-white' threshold.
    Returns: mask (uint8) where foreground=255, background=0
    """
    h, w = bgr.shape[:2]

    if corner_sample:
        # Sample small corner patches to estimate background color.
        patch = max(8, min(h, w) // 50)
        corners = [
            bgr[0:patch, 0:patch],
            bgr[0:patch, w - patch:w],
            bgr[h - patch:h, 0:patch],
            bgr[h - patch:h, w - patch:w],
        ]
        bg = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)

        # Distance threshold in RGB-space (simple and fast).
        diff = np.linalg.norm(bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
        # Convert "white_threshold" (0..255) to a distance threshold.
        # Higher white_threshold => stricter background => smaller diff threshold.
        # This mapping is heuristic but works well in practice.
        dist_thr = max(10.0, (255 - float(white_threshold)) * 1.2)
        bg_mask = (diff <= dist_thr).astype(np.uint8) * 255
    else:
        # Simple: treat pixels that are "very bright" as background.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bg_mask = (gray >= white_threshold).astype(np.uint8) * 255

    fg_mask = cv2.bitwise_not(bg_mask)
    return fg_mask


def _clean_mask(mask: np.ndarray, smooth: bool = True, blur_ksize: int = 7) -> np.ndarray:
    """Clean up a binary mask."""
    if smooth:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # Re-binarize
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to remove small holes & jaggies
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    """Return largest contour in mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found. Try changing white_threshold.")
    largest = max(contours, key=cv2.contourArea)
    return largest


def _smooth_contour(contour: np.ndarray, epsilon_ratio: float = 0.0015) -> np.ndarray:
    """
    Smooth a contour using approxPolyDP.
    epsilon_ratio is relative to arc length. Lower => more detail, higher => smoother.
    """
    arc = cv2.arcLength(contour, True)
    eps = max(1.0, arc * float(epsilon_ratio))
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx


def _draw_stroke(
    contour: np.ndarray,
    canvas_size: Tuple[int, int],
    thickness: int = 6,
) -> np.ndarray:
    """Draw a single contour as stroke on white background."""
    w, h = canvas_size
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.drawContours(
        out,
        [contour],
        contourIdx=-1,
        color=(0, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,  # IMPORTANT: anti-aliased stroke
    )
    return out


def _png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Could not encode PNG.")
    return buf.tobytes()


def _pdf_bytes_from_png(png_bytes: bytes) -> bytes:
    """
    Simple PDF: embeds PNG on a 1-page PDF (vector comes later).
    This is acceptable for now; next step is true vector PDF.
    """
    img = ImageReader(io.BytesIO(png_bytes))

    # Try to get image size
    iw, ih = img.getSize()

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=(iw, ih))
    c.drawImage(img, 0, 0, width=iw, height=ih, mask="auto")
    c.showPage()
    c.save()

    return pdf_buf.getvalue()


# -------------------------
# API
# -------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


@app.post("/stroke/preview")
async def stroke_preview(
    file: UploadFile = File(...),
    thickness: int = Query(6, ge=1, le=40),
    white_threshold: int = Query(235, ge=150, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    upscale: int = Query(4, ge=1, le=8),
    corner_sample: bool = Query(True),
):
    """
    Returns a PNG preview with a clean, anti-aliased stroke outline.
    - white_threshold: higher => requires whiter background, lower => more tolerant
    - corner_sample: estimates background color from corners (recommended)
    - upscale: internal scale factor for smoother edges
    """
    bgr = _read_upload_to_bgr(file)

    # Internal upscale (makes edges smoother)
    if upscale > 1:
        bgr = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    mask = _auto_white_mask_from_background(bgr, white_threshold=white_threshold, corner_sample=corner_sample)
    mask = _clean_mask(mask, smooth=smooth, blur_ksize=7)

    contour = _largest_contour(mask)
    contour = _smooth_contour(contour, epsilon_ratio=epsilon_ratio)

    h, w = mask.shape[:2]
    out = _draw_stroke(contour, canvas_size=(w, h), thickness=max(1, int(thickness * upscale)))

    # Downscale back to “nice” preview size
    if upscale > 1:
        out = cv2.resize(out, (w // upscale, h // upscale), interpolation=cv2.INTER_AREA)

    return Response(content=_png_bytes(out), media_type="image/png")


@app.post("/stroke/pdf")
async def stroke_pdf(
    file: UploadFile = File(...),
    thickness: int = Query(6, ge=1, le=40),
    white_threshold: int = Query(235, ge=150, le=255),
    smooth: bool = Query(True),
    epsilon_ratio: float = Query(0.0015, ge=0.0003, le=0.02),
    upscale: int = Query(4, ge=1, le=8),
    corner_sample: bool = Query(True),
):
    """
    Returns a PDF (currently image-based PDF).
    Next step: true vector PDF (paths).
    """
    # Reuse preview logic to generate a clean PNG first
    bgr = _read_upload_to_bgr(file)

    if upscale > 1:
        bgr = cv2.resize(bgr, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    mask = _auto_white_mask_from_background(bgr, white_threshold=white_threshold, corner_sample=corner_sample)
    mask = _clean_mask(mask, smooth=smooth, blur_ksize=7)

    contour = _largest_contour(mask)
    contour = _smooth_contour(contour, epsilon_ratio=epsilon_ratio)

    h, w = mask.shape[:2]
    out = _draw_stroke(contour, canvas_size=(w, h), thickness=max(1, int(thickness * upscale)))

    if upscale > 1:
        out = cv2.resize(out, (w // upscale, h // upscale), interpolation=cv2.INTER_AREA)

    png = _png_bytes(out)
    pdf = _pdf_bytes_from_png(png)

    return Response(content=pdf, media_type="application/pdf")
