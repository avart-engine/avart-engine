from __future__ import annotations

import os
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


SERVICE_NAME = "avart-engine"

app = FastAPI(title=SERVICE_NAME, version="0.1.0")

# (Praktisk når du senere kalder API'et fra avart.dk / one.com)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # stram dette senere
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME}


def _read_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into OpenCV BGR image."""
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a JPG/PNG.")
    return img


def _auto_threshold_for_profile(gray: np.ndarray) -> np.ndarray:
    """
    Create a binary mask where the subject is white (255) and background black (0).
    We assume a bright/white background is best, but we also auto-invert if needed.
    """
    # Reduce noise a bit
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine if background is mostly white; if not, invert
    # corners are usually background
    h, w = th.shape[:2]
    corners = np.concatenate(
        [
            th[0:10, 0:10].ravel(),
            th[0:10, w - 10 : w].ravel(),
            th[h - 10 : h, 0:10].ravel(),
            th[h - 10 : h, w - 10 : w].ravel(),
        ]
    )
    bg_is_white = (corners.mean() > 127)

    # If background is white, subject tends to be darker -> threshold likely makes bg white.
    # We want subject=white, bg=black -> invert when bg is white.
    if bg_is_white:
        th = cv2.bitwise_not(th)

    # Clean small specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return th


def _largest_external_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns the largest external contour (by area).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _smooth_contour(cnt: np.ndarray, window: int = 15) -> np.ndarray:
    """
    Simple moving-average smoothing over contour points.
    Keeps shape but reduces jaggedness. Works well for "stroke".
    """
    pts = cnt[:, 0, :].astype(np.float32)
    n = len(pts)
    if n < max(10, window):
        return cnt

    window = max(5, window)
    if window % 2 == 0:
        window += 1

    pad = window // 2
    pts_pad = np.vstack([pts[-pad:], pts, pts[:pad]])

    smoothed = np.empty_like(pts)
    for i in range(n):
        seg = pts_pad[i : i + window]
        smoothed[i] = seg.mean(axis=0)

    smoothed = np.round(smoothed).astype(np.int32).reshape(-1, 1, 2)
    return smoothed


def _render_stroke_png(
    bgr: np.ndarray,
    line_thickness_px: int = 4,
    simplify_eps_ratio: float = 0.0015,
    smooth_window: int = 17,
    pad_px: int = 10,
) -> bytes:
    """
    Make a white canvas with a single black outer contour (profile stroke).
    Returns PNG bytes.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = _auto_threshold_for_profile(gray)
    cnt = _largest_external_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 2000:
        raise HTTPException(
            status_code=400,
            detail="Could not detect a clear silhouette. Use white background and darker subject.",
        )

    # Slight simplification (fewer points)
    arc = cv2.arcLength(cnt, True)
    eps = max(1.0, simplify_eps_ratio * arc)
    cnt = cv2.approxPolyDP(cnt, eps, True)

    # Smooth the contour for nicer lines
    cnt = _smooth_contour(cnt, window=smooth_window)

    # Add padding so stroke doesn't touch edges
    out_h, out_w = h + 2 * pad_px, w + 2 * pad_px
    out = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    cnt2 = cnt.copy()
    cnt2[:, 0, 0] += pad_px
    cnt2[:, 0, 1] += pad_px

    # Draw stroke
    cv2.drawContours(
        out,
        [cnt2],
        contourIdx=-1,
        color=(0, 0, 0),
        thickness=int(line_thickness_px),
        lineType=cv2.LINE_AA,
    )

    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG.")
    return buf.tobytes()


@app.post("/stroke/preview")
async def stroke_preview(
    file: UploadFile = File(...),
    thickness: int = 4,
):
    """
    Upload a JPG/PNG profile photo and get a stroke-only preview (PNG).
    - thickness: line thickness in pixels (preview)
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    bgr = _read_upload_to_bgr(content)

    # Clamp thickness (preview only)
    thickness = int(max(1, min(thickness, 12)))

    png_bytes = _render_stroke_png(
        bgr=bgr,
        line_thickness_px=thickness,
        simplify_eps_ratio=0.0015,
        smooth_window=17,
        pad_px=10,
    )
    return Response(content=png_bytes, media_type="image/png")


@app.get("/")
def root():
    return JSONResponse({"hello": "world"})


# Render will run: uvicorn main:app --host 0.0.0.0 --port $PORT
# If you ever run locally:
#   uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
