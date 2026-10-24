from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config
from .inference import heatmap_png, load_predictor, preprocess

STATIC = Path(__file__).resolve().parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    state["predictor"] = load_predictor(config.FRAMEWORK, config.MODEL_PATH, config.IMAGE_SIZE)
    state["load_s"] = time.perf_counter() - t0
    print(f"[{config.TEAM_ID}] {config.FRAMEWORK} model loaded from {config.MODEL_PATH} "
          f"in {state['load_s']:.1f}s")
    yield
    state.clear()


app = FastAPI(
    title="Brain MRI Tumour Classifier",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,           # replaced below by an offline Swagger UI
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    framework: str = Field(examples=[config.FRAMEWORK])
    classes: List[str] = Field(examples=[config.CLASS_NAMES])


class PredictResponse(BaseModel):
    framework: str = Field(examples=[config.FRAMEWORK])
    label: str = Field(examples=["glioma"])
    confidence: float = Field(examples=[0.94], ge=0, le=1)
    probabilities: Dict[str, float] = Field(
        examples=[{"glioma": 0.94, "meningioma": 0.04, "notumor": 0.01, "pituitary": 0.01}])
    latency_ms: int = Field(examples=[85], description="Server-side preprocessing + inference time")


class ExplainResponse(BaseModel):
    label: str
    heatmap_png: str = Field(description="Base64 RGBA PNG, IMAGE_SIZE x IMAGE_SIZE, to overlay on the scan")
    latency_ms: int


class ErrorResponse(BaseModel):
    detail: str


ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty or corrupted image"},
    413: {"model": ErrorResponse, "description": "File too large"},
    415: {"model": ErrorResponse, "description": "Not a JPEG or PNG image"},
}


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty. Send a JPEG or PNG image in the 'file' field.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The file is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.verify()                                  # detects truncated / corrupted files
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(415, f"'{file.filename}' is not an image. Send a JPEG or PNG file.")
    if fmt not in config.ALLOWED_FORMATS:
        raise HTTPException(415, f"'{file.filename}' is a {fmt} image. Only JPEG and PNG are accepted.")
    try:
        img = Image.open(io.BytesIO(data))            # re-open: verify() leaves the image unusable
        img.load()
    except Exception:
        raise HTTPException(400, f"'{file.filename}' could not be decoded. The file may be corrupted.")
    return img


@app.get("/health", response_model=HealthResponse, tags=["service"], summary="Service status")
def health():
    return {"status": "ok", "framework": config.FRAMEWORK, "classes": config.CLASS_NAMES}


@app.post("/predict", response_model=PredictResponse, responses=ERRORS, tags=["service"],
          summary="Classify a brain MRI image")
async def predict(file: UploadFile = File(..., description="Brain MRI image (JPEG or PNG)")):
    img = await read_image(file)
    predictor = state["predictor"]

    def run():
        t0 = time.perf_counter()
        x = preprocess(img, config.IMAGE_SIZE)
        probs = predictor.predict(x)
        return probs, (time.perf_counter() - t0) * 1000

    probs, ms = await run_in_threadpool(run)
    k = int(probs.argmax())
    return {
        "framework": config.FRAMEWORK,
        "label": config.CLASS_NAMES[k],
        "confidence": round(float(probs[k]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(config.CLASS_NAMES, probs)},
        "latency_ms": int(round(ms)),
    }


@app.post("/explain", response_model=ExplainResponse, responses=ERRORS, tags=["interpretability"],
          summary="Grad-CAM attention map for the predicted class")
async def explain(file: UploadFile = File(..., description="Brain MRI image (JPEG or PNG)")):
    img = await read_image(file)
    predictor = state["predictor"]

    def run():
        t0 = time.perf_counter()
        x = preprocess(img, config.IMAGE_SIZE)
        k = int(predictor.predict(x).argmax())
        heat = predictor.gradcam(x, k)
        return k, heatmap_png(heat), (time.perf_counter() - t0) * 1000

    k, png, ms = await run_in_threadpool(run)
    return {"label": config.CLASS_NAMES[k], "heatmap_png": base64.b64encode(png).decode(),
            "latency_ms": int(round(ms))}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/docs", include_in_schema=False)
def docs():
    # Swagger UI served from local files so /docs works without internet.
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} API docs",
        swagger_js_url="/static/vendor/swagger-ui-bundle.js",
        swagger_css_url="/static/vendor/swagger-ui.css",
        swagger_favicon_url="/static/favicon.svg",
    )


def _mark_uploads_as_binary(node):
    """Newer FastAPI versions describe uploaded files as {"contentMediaType": "application/octet-stream"}
    (OpenAPI 3.1). The offline Swagger UI only shows a "Choose File" button for {"format": "binary"},
    so add that wherever an upload appears in the schema."""
    if isinstance(node, dict):
        if node.get("type") == "string" and "contentMediaType" in node:
            node["format"] = "binary"
        for value in node.values():
            _mark_uploads_as_binary(value)
    elif isinstance(node, list):
        for value in node:
            _mark_uploads_as_binary(value)


def openapi_with_file_upload():
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version,
                             description=app.description, routes=app.routes)
        _mark_uploads_as_binary(schema)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi_with_file_upload


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Inference failed: {type(exc).__name__}"})
