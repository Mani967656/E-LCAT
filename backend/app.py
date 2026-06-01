from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent

MODEL_PATH = Path(os.environ.get("LCAT_MODEL_PATH", str(REPO_ROOT / "best_lcat_model (1).h5")))
LABELS_PATH = Path(os.environ.get("LCAT_LABELS_PATH", str(REPO_ROOT / "labels.json")))
RECOMMENDATIONS_PATH = Path(
    os.environ.get("LCAT_RECOMMENDATIONS_PATH", str(REPO_ROOT / "recommendations.json"))
)

UPLOAD_DIR = REPO_ROOT / "uploads"
STATIC_DIR = REPO_ROOT / "static"
TEMPLATES_DIR = REPO_ROOT / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LCAT Leaf Disease Classifier")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ModelBundle:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.input_hw: tuple[int, int] | None = None
        self.labels: list[str] | None = None
        self.recommendations: dict[str, list[str]] | None = None


bundle = ModelBundle()


def _lazy_import_tf():
    import tensorflow as tf  # noqa: WPS433 (runtime import by design)

    return tf


def _load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_input_hw(model: Any) -> tuple[int, int]:
    shape = getattr(model, "input_shape", None)
    if not shape:
        return (224, 224)
    # shape can be (None, H, W, C) or (None, C, H, W)
    dims = [d for d in shape if isinstance(d, int)]
    if len(dims) >= 3:
        # Heuristic: if last is 1/3/4 then channel-last
        if dims[-1] in (1, 3, 4):
            return (dims[-3], dims[-2])
        # Otherwise assume channel-first
        if dims[0] in (1, 3, 4) and len(dims) >= 3:
            return (dims[1], dims[2])
    return (224, 224)


def _infer_num_classes(model: Any) -> int:
    out_shape = getattr(model, "output_shape", None)
    if not out_shape:
        return 0
    if isinstance(out_shape, list) and out_shape:
        out_shape = out_shape[0]
    if isinstance(out_shape, tuple) and len(out_shape) >= 2 and isinstance(out_shape[-1], int):
        return out_shape[-1]
    return 0


def ensure_model_loaded() -> None:
    if bundle.model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        tf = _lazy_import_tf()

        # Keras 3 can choke on some legacy-saved configs that include extra keys
        # like Dense.quantization_config=None. We patch deserialization to ignore
        # these no-op fields so the model can load.
        try:
            import keras  # noqa: WPS433

            _orig_dense_from_config = keras.layers.Dense.from_config

            def _dense_from_config_patched(cls, config):  # type: ignore[no-untyped-def]
                if isinstance(config, dict):
                    config = dict(config)
                    config.pop("quantization_config", None)
                return _orig_dense_from_config.__func__(cls, config)  # type: ignore[attr-defined]

            # Only patch once
            if getattr(keras.layers.Dense.from_config, "__name__", "") != "_dense_from_config_patched":
                keras.layers.Dense.from_config = classmethod(_dense_from_config_patched)  # type: ignore[assignment]
        except Exception:
            pass

        custom_objects: dict[str, Any] = {}
        try:
            from backend.custom_layers import get_custom_objects  # noqa: WPS433

            custom_objects = get_custom_objects() or {}
        except Exception:
            custom_objects = {}

        try:
            bundle.model = tf.keras.models.load_model(
                str(MODEL_PATH),
                compile=False,
                custom_objects=custom_objects,
            )
        except Exception as e:
            msg = str(e)
            if "Unknown layer" in msg or "Unknown object" in msg:
                raise RuntimeError(
                    "Model load failed because it uses a custom Keras layer that is not available at runtime. "
                    "Add the layer implementation to `backend/custom_layers.py` (and return it from "
                    "`get_custom_objects()`), then restart the server. Original error: "
                    + msg
                ) from e
            raise
        bundle.input_hw = _infer_input_hw(bundle.model)

    # Always reload labels/recommendations from disk so changes take effect
    # without needing a restart.
    labels_json = _load_json_if_exists(LABELS_PATH)
    if isinstance(labels_json, list) and all(isinstance(x, str) for x in labels_json):
        bundle.labels = labels_json
    else:
        n = _infer_num_classes(bundle.model)
        bundle.labels = [f"class_{i}" for i in range(n)] if n > 0 else ["class_0"]

    rec_json = _load_json_if_exists(RECOMMENDATIONS_PATH)
    if isinstance(rec_json, dict):
        bundle.recommendations = {
            str(k): [str(x) for x in (v if isinstance(v, list) else [v])] for k, v in rec_json.items()
        }
    else:
        bundle.recommendations = {}


@app.get("/info")
def info():
    ensure_model_loaded()
    return {
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH),
        "labels_exists": LABELS_PATH.exists(),
        "labels_len": len(bundle.labels or []),
        "labels_head": (bundle.labels or [])[:5],
        "output_shape": getattr(bundle.model, "output_shape", None) if bundle.model is not None else None,
    }


def preprocess_image(img: Image.Image, hw: tuple[int, int], mode: str) -> np.ndarray:
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = img.resize((hw[1], hw[0]))
    arr = np.asarray(img).astype("float32")

    mode = (mode or "0_1").strip().lower()
    if mode in ("0_1", "0-1", "01"):
        arr = arr / 255.0
    elif mode in ("-1_1", "-1-1", "m1_1"):
        arr = (arr / 127.5) - 1.0
    elif mode in ("0_255", "0-255", "raw"):
        # leave as 0..255
        pass
    else:
        # fallback
        arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)
    return arr


def _predict_probs(arr: np.ndarray) -> np.ndarray:
    ensure_model_loaded()
    assert bundle.model is not None

    preds = bundle.model.predict(arr, verbose=0)
    if isinstance(preds, list):
        preds = preds[0]
    preds = np.asarray(preds).reshape(-1)
    if preds.size == 0:
        preds = np.array([1.0], dtype="float32")

    probs = preds.astype("float32")
    if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(np.sum(probs), 1.0, atol=1e-2):
        e = np.exp(probs - np.max(probs))
        probs = e / np.sum(e)
    return probs


def predict(arr: np.ndarray) -> dict[str, Any]:
    ensure_model_loaded()
    assert bundle.labels is not None

    probs = _predict_probs(arr)

    topk = int(min(3, probs.size))
    top_idx = probs.argsort()[::-1][:topk]

    labels = bundle.labels
    if len(labels) < probs.size:
        labels = labels + [f"class_{i}" for i in range(len(labels), probs.size)]

    best_i = int(top_idx[0])
    best_label = labels[best_i]
    best_conf = float(probs[best_i])

    top = [{"label": labels[int(i)], "confidence": float(probs[int(i)])} for i in top_idx]
    recs = (bundle.recommendations or {}).get(best_label, [])
    if not recs:
        recs = [
            "Remove heavily infected leaves and dispose away from the field.",
            "Avoid overhead irrigation; keep foliage dry when possible.",
            "Apply a suitable fungicide/bactericide as per local agricultural guidance.",
            "Ensure balanced nutrition and spacing to improve airflow.",
        ]
    recs = recs[:2]

    return {
        "label": best_label,
        "confidence": best_conf,
        "top": top,
        "recommendations": recs,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_path": str(MODEL_PATH),
            "labels_path": str(LABELS_PATH),
            "recommendations_path": str(RECOMMENDATIONS_PATH),
        },
    )


@app.post("/predict")
async def predict_route(
    request: Request,
    image: UploadFile = File(...),
    crop: str = Form(default=""),
    preprocess: str = Form(default="auto"),
):
    try:
        ensure_model_loaded()
        assert bundle.input_hw is not None
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": str(e),
                "result": None,
                "image_url": None,
                "crop": crop,
            },
            status_code=500,
        )

    import io  # noqa: WPS433 (runtime import by design)

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "error": "Could not read the uploaded image. Please upload a valid JPG/PNG.",
                "result": None,
                "image_url": None,
                "crop": crop,
            },
            status_code=400,
        )

    uid = uuid.uuid4().hex
    ext = Path(image.filename or "upload.jpg").suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    saved_path = UPLOAD_DIR / f"{uid}{ext}"
    img_rgb = ImageOps.exif_transpose(img).convert("RGB")
    img_rgb.save(saved_path, quality=95)

    # Try common preprocessing modes if requested.
    modes = [preprocess] if preprocess != "auto" else ["0_1", "-1_1", "0_255"]
    best_mode = modes[0]
    best_probs: np.ndarray | None = None
    best_score = -1.0
    for m in modes:
        arr = preprocess_image(img_rgb, bundle.input_hw, m)
        probs = _predict_probs(arr)
        score = float(np.max(probs))
        if score > best_score:
            best_score = score
            best_probs = probs
            best_mode = m

    assert best_probs is not None
    probs = best_probs
    topk = int(min(3, probs.size))
    top_idx = probs.argsort()[::-1][:topk]
    labels = bundle.labels or []
    if len(labels) < probs.size:
        labels = labels + [f"class_{i}" for i in range(len(labels), probs.size)]
    best_i = int(top_idx[0])
    best_label = labels[best_i]
    recs = (bundle.recommendations or {}).get(best_label, [])
    if not recs:
        recs = [
            "Remove heavily infected leaves and dispose away from the field.",
            "Avoid overhead irrigation; keep foliage dry when possible.",
            "Apply a suitable fungicide/bactericide as per local agricultural guidance.",
            "Ensure balanced nutrition and spacing to improve airflow.",
        ]
    recs = recs[:2]

    result = {
        "label": best_label,
        "confidence": float(probs[best_i]),
        "top": [{"label": labels[int(i)], "confidence": float(probs[int(i)])} for i in top_idx],
        "recommendations": recs,
        "preprocess": best_mode,
    }

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "error": None,
            "result": result,
            "image_url": f"/uploads/{saved_path.name}",
            "crop": crop,
            "preprocess": best_mode,
        },
    )


@app.get("/health")
def health():
    return {"ok": True}

