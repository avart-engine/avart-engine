from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import io
import numpy as np
import cv2
from PIL import Image
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


app = FastAPI(title="avart-engine", version="0.1.0")

# ✅ Allow calls from your One.com frontend (and local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://avart.dk",
        "https://www.avart.dk",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"hello": "world"}

@app.get("/health")
def health():
    return {"ok": True}


def _read_upload_to_bgr(upload: UploadFile) -> np.ndarray:
    """
    Read uploaded image into OpenCV BGR numpy array.
    Supports JPG/PNG/WEBP etc.
    """
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _stroke_from_image(
    bgr: np.ndarray,
    stroke_px: int = 6,
    canny1: int = 60,
    canny2: int = 140,
    smooth: int = 5,
) -> np.ndarray:
    """
    Create an OUTER CONTOUR stroke on transparent background.

    Returns RGBA image (uint8) where the stroke is black and alpha=255,
    background alpha=0.
    """
    # Convert to gray and smooth
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if smooth and smooth > 1:
        k = smooth if smooth % 2 == 1 else smooth + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Detect edges
    edges = cv2.Canny(gray, canny1, canny2)

    # Close small gaps and make one clean contour
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours, keep only the largest (outer contour)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if not contours:
        return rgba

    # Biggest contour by area
    c = max(contours, key=cv2.contourArea)

    # Optional: simplify contour a bit for cleaner line
    peri = cv2.arcLength(c, True)
    eps = 0.003 * peri
    c_simpl = cv2.approxPolyDP(c, eps, True)

    # Draw black stroke on alpha channel
    # lineType=cv2.LINE_AA gives smooth stroke
    cv2.drawContours(rgba, [c_simpl], -1, (0, 0, 0, 255), thickness=stroke_px, lineType=cv2.LINE_AA)

    return rgba


@app.post("/stroke/preview", summary="Upload -> returns PNG (transparent) with outer stroke")
async def stroke_preview(
    file: UploadFile = File(...),
    stroke_px: int = Form(6),
    canny1: int = Form(60),
    canny2: int = Form(140),
    smooth: int = Form(5),
):
    try:
        bgr = _read_upload_to_bgr(file)
        rgba = _stroke_from_image(bgr, stroke_px=stroke_px, canny1=canny1, canny2=canny2, smooth=smooth)

        # Encode PNG
        ok, png = cv2.imencode(".png", rgba)
        if not ok:
            return JSONResponse({"error": "Could not encode png"}, status_code=500)

        return Response(content=png.tobytes(), media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/stroke/pdf", summary="Upload -> returns 1-page PDF (stroke PNG placed on page)")
async def stroke_pdf(
    file: UploadFile = File(...),
    page_w_mm: int = Form(500),
    page_h_mm: int = Form(700),
    dpi: int = Form(300),
    stroke_px: int = Form(6),
    canny1: int = Form(60),
    canny2: int = Form(140),
    smooth: int = Form(5),
):
    """
    Very simple PDF: place the transparent stroke PNG on a page.
    Later we’ll plug in your size rules, margins, logo, top text etc.
    """
    try:
        bgr = _read_upload_to_bgr(file)
        rgba = _stroke_from_image(bgr, stroke_px=stroke_px, canny1=canny1, canny2=canny2, smooth=smooth)

        # Convert RGBA -> PNG bytes via PIL (keeps transparency nicely)
        pil = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)

        # mm -> points
        # 1 inch = 25.4mm, 1 point = 1/72 inch
        def mm_to_pt(mm: float) -> float:
            return (mm / 25.4) * 72.0

        page_w_pt = mm_to_pt(page_w_mm)
        page_h_pt = mm_to_pt(page_h_mm)

        pdf_buf = io.BytesIO()
        c = pdf_canvas.Canvas(pdf_buf, pagesize=(page_w_pt, page_h_pt))

        # For now: fill whole page (later we respect margins)
        img_reader = ImageReader(buf)
        c.drawImage(img_reader, 0, 0, width=page_w_pt, height=page_h_pt, mask='auto')

        c.showPage()
        c.save()

        pdf_buf.seek(0)
        return Response(content=pdf_buf.read(), media_type="application/pdf")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
