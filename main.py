from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io
import os
import tempfile

import numpy as np
import cv2
from rembg import remove, new_session

from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


app = FastAPI(title="avart-engine")


# ---------------------------------
# DESIGN SETTINGS
# ---------------------------------

MAX_DIMENSION = 1600
REMBG_MODEL = "u2net"

BG_COLOR = "#e9e3db"

PAGE_W_MM = 500
PAGE_H_MM = 700

TOP_BAND_MM = 115

TITLE_FONT = "Helvetica-Bold"
TITLE_FONT_SIZE = 35

LOGO_WIDTH_MM = 50
LOGO_BOTTOM_MM = 50

SILHOUETTE_TOP_GAP_MM = 18
SILHOUETTE_SIDE_MARGIN_MM = 32
SILHOUETTE_BOTTOM_GAP_MM = 0

DEFAULT_STROKE_WIDTH = 3.5

# Sæt denne hvis du har et rigtigt logo liggende
LOGO_PATH = None
# eksempel:
# LOGO_PATH = "assets/avart-logo.png"


_rembg_session = None


# ---------------------------------
# HELPERS
# ---------------------------------

def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session(REMBG_MODEL)
    return _rembg_session


def resize_if_needed_rgba(img: np.ndarray, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_dimension:
        return img

    scale = max_dimension / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def remove_background_if_needed(upload: UploadFile, max_dimension: int = MAX_DIMENSION) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise ValueError("Empty file")

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Could not decode image")

    # Hvis billedet allerede har alpha og faktisk transparency
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < 250):
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            rgba = resize_if_needed_rgba(rgba, max_dimension=max_dimension)

            # ekstra transparent bund
            bottom_pad = 180
            rgba = cv2.copyMakeBorder(
                rgba,
                0,
                bottom_pad,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0, 0),
            )
            return rgba

    # Resize før rembg
    max_input_size = 1600
    h, w = img.shape[:2]
    scale = min(1.0, max_input_size / max(h, w))

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("Could not encode resized image")

        data = buffer.tobytes()

    # Fjern baggrund
    output = remove(data, session=get_rembg_session())

    arr_out = np.frombuffer(output, np.uint8)
    img_out = cv2.imdecode(arr_out, cv2.IMREAD_UNCHANGED)

    if img_out is None:
        raise ValueError("Background removal failed")

    if len(img_out.shape) == 3 and img_out.shape[2] == 3:
        alpha = np.full((img_out.shape[0], img_out.shape[1], 1), 255, dtype=np.uint8)
        img_out = np.concatenate([img_out, alpha], axis=2)

    if len(img_out.shape) != 3 or img_out.shape[2] != 4:
        raise ValueError("Background removal did not return RGBA")

    # ekstra transparent bund
    bottom_pad = 180
    img_out = cv2.copyMakeBorder(
        img_out,
        0,
        bottom_pad,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )

    rgba = cv2.cvtColor(img_out, cv2.COLOR_BGRA2RGBA)
    return resize_if_needed_rgba(rgba, max_dimension=max_dimension)


def alpha_to_mask(rgba: np.ndarray, alpha_threshold: int = 1) -> np.ndarray:
    alpha = rgba[:, :, 3]
    return (alpha > alpha_threshold).astype(np.uint8) * 255


def get_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found")
    return max(contours, key=cv2.contourArea)


def smooth_contour(contour: np.ndarray, epsilon_ratio: float = 0.002):
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * peri
    return cv2.approxPolyDP(contour, epsilon, True)


def contour_to_svg(contour: np.ndarray, width: int, height: int, stroke_width: float = DEFAULT_STROKE_WIDTH) -> str:
    pts = contour[:, 0, :]
    path = "M " + " L ".join([f"{x},{y}" for x, y in pts])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <path d="{path}" fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />
</svg>"""
    return svg


def draw_svg_on_pdf(
    pdf_canvas,
    svg_string: str,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
):
    tmp_svg = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
    try:
        tmp_svg.write(svg_string.encode("utf-8"))
        tmp_svg.close()

        drawing = svg2rlg(tmp_svg.name)
        if drawing is None:
            raise ValueError("Could not load SVG drawing")

        raw_w = drawing.width
        raw_h = drawing.height

        if raw_w <= 0 or raw_h <= 0:
            raise ValueError("Invalid SVG size")

        scale = min(max_width / raw_w, max_height / raw_h)

        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)

        pdf_canvas.saveState()
        pdf_canvas.translate(x, y)
        renderPDF.draw(drawing, pdf_canvas, 0, 0)
        pdf_canvas.restoreState()

    finally:
        try:
            os.unlink(tmp_svg.name)
        except Exception:
            pass


def generate_poster_pdf(
    svg_string: str,
    name: str,
    bg_color: str = BG_COLOR,
    logo_path: str | None = LOGO_PATH,
) -> bytes:
    buffer = io.BytesIO()

    width = PAGE_W_MM * mm
    height = PAGE_H_MM * mm

    top_band_h = TOP_BAND_MM * mm
    logo_width = LOGO_WIDTH_MM * mm
    logo_bottom = LOGO_BOTTOM_MM * mm

    side_margin = SILHOUETTE_SIDE_MARGIN_MM * mm
    top_gap = SILHOUETTE_TOP_GAP_MM * mm
    bottom_gap = SILHOUETTE_BOTTOM_GAP_MM * mm

    c = canvas.Canvas(buffer, pagesize=(width, height))

    # background
    c.setFillColor(colors.HexColor(bg_color))
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # title
    c.setFillColor(colors.black)
    c.setFont(TITLE_FONT, TITLE_FONT_SIZE)
    title_y = height - (top_band_h / 2) - (TITLE_FONT_SIZE * 0.35)
    c.drawCentredString(width / 2, title_y, name)

    # silhouette area
    silhouette_top_y = height - top_band_h - top_gap
    silhouette_bottom_y = logo_bottom + (18 * mm) + bottom_gap
    silhouette_height = silhouette_top_y - silhouette_bottom_y
    silhouette_width = width - (2 * side_margin)

    draw_svg_on_pdf(
        c,
        svg_string=svg_string,
        x=side_margin,
        y=silhouette_bottom_y,
        max_width=silhouette_width,
        max_height=silhouette_height,
    )

    # logo
    if logo_path and os.path.exists(logo_path):
        try:
            logo = ImageReader(logo_path)
            lw, lh = logo.getSize()
            logo_h = logo_width * (lh / lw)

            logo_x = (width - logo_width) / 2
            logo_y = logo_bottom

            c.drawImage(
                logo,
                logo_x,
                logo_y,
                width=logo_width,
                height=logo_h,
                mask="auto",
            )
        except Exception:
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(width / 2, logo_bottom + 6, "avart")
    else:
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, logo_bottom + 6, "avart")

    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------------------------
# API
# ---------------------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "avart-engine"}


@app.post(
    "/poster/pdf",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {
                "application/pdf": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "description": "PDF file",
        }
    },
)
async def poster_pdf(
    file: UploadFile = File(...),
    name: str = Query("Test"),
    stroke_width: float = Query(DEFAULT_STROKE_WIDTH),
):
    try:
        rgba = remove_background_if_needed(file, max_dimension=MAX_DIMENSION)

        mask = alpha_to_mask(rgba)
        contour = get_contour(mask)
        contour = smooth_contour(contour)

        h, w = rgba.shape[:2]
        svg = contour_to_svg(contour, w, h, stroke_width)

        pdf_bytes = generate_poster_pdf(
            svg,
            name=name,
            bg_color=BG_COLOR,
            logo_path=LOGO_PATH,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{name}.pdf"'
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
