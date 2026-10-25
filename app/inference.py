from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

# The service runs on CPU only; hide GPUs and quieten TensorFlow before import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent     


class ModelFileError(RuntimeError):
    """The model file is missing or is not a real model."""


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise ModelFileError(f"Model file not found: {p}")
    with open(p, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise ModelFileError(
            f"{p.name} is a Git LFS pointer ({p.stat().st_size} bytes), not the trained model. "
            "Download the real file (git lfs pull, or the 'Download raw file' button on GitHub) "
            "and put it in the model/ folder."
        )
    return p


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """Exactly the notebooks' serve_pre: HWC float32, values 0..255."""
    return np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32")



_STOPS = np.array([          # transparent cool -> warm, alpha grows with attention
    [0.00,  40,  60, 140,   0],
    [0.35,  40, 140, 200,  90],
    [0.60, 250, 210,  60, 170],
    [0.80, 245, 120,  40, 210],
    [1.00, 220,  30,  45, 235],
], dtype="float32")


def heatmap_png(heat: np.ndarray) -> bytes:
    """Map a 0..1 heatmap to an RGBA PNG (low attention fades to transparent)."""
    h = np.clip(heat, 0, 1).ravel()
    rgba = np.stack([np.interp(h, _STOPS[:, 0], _STOPS[:, c]) for c in range(1, 5)], axis=-1)
    img = Image.fromarray(rgba.reshape(*heat.shape, 4).astype("uint8"), "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()



class TensorFlowPredictor:
    framework = "tensorflow"

    def __init__(self, path: Path, size: int):
        import tensorflow as tf
        import keras
        self.tf, self.keras, self.size = tf, keras, size
        self.model = keras.models.load_model(path, compile=False)
        self._cam = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.model(x[None], training=False)          # softmax already in the model
        return np.asarray(probs)[0].astype("float64")

    # Same recipe as make_gradcam() in tensorflow_T10.ipynb, scored before softmax (see PyTorch note)
    def _build_cam(self):
        keras, tf = self.keras, self.tf
        m = self.model
        base = next((l for l in m.layers if l.name.endswith("_base")), None) or next(
            l for l in m.layers if isinstance(l, keras.Model) and l.name != "augment")
        last = next(l.name for l in reversed(base.layers) if len(l.output.shape) == 4)
        feat = keras.Model(base.input, [base.get_layer(last).output, base.output])
        prep, gap, drop, out = (m.get_layer(n) for n in ["prep", "gap", "drop", "out"])

        def cam(img: np.ndarray, k: int):
            x = prep(tf.convert_to_tensor(img[None], tf.float32))
            with tf.GradientTape() as tape:
                conv, f = feat(x, training=False)
                h = drop(gap(f), training=False)
                score = tf.matmul(h, out.kernel)[:, k] + out.bias[k]   # pre-softmax score
            g = tape.gradient(score, conv)
            w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
            c = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0]
            c = c / (tf.reduce_max(c) + 1e-8)
            return tf.image.resize(c[..., None], (self.size, self.size))[..., 0].numpy()
        return cam

    def gradcam(self, x: np.ndarray, k: int) -> np.ndarray:
        if self._cam is None:
            self._cam = self._build_cam()
        return self._cam(x, k)


class PyTorchPredictor:
    framework = "pytorch"

    def __init__(self, path: Path, size: int):
        import torch
        import torch.nn.functional as F
        self.torch, self.F, self.size = torch, F, size
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.model = torch.jit.load(str(path), map_location="cpu").eval()

    def _batch(self, x: np.ndarray):
        return self.torch.from_numpy(x[None]).permute(0, 3, 1, 2).contiguous()   # NHWC -> NCHW

    def predict(self, x: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            logits = self.model(self._batch(x))                  # model returns logits
            return self.F.softmax(logits, 1)[0].double().numpy()

    
    def gradcam(self, x: np.ndarray, k: int) -> np.ndarray:
        torch, F, m = self.torch, self.F, self.model
        with torch.enable_grad():
            z = (self._batch(x) - m.mean) / m.std              # == model.prep (ImageNet mean/std)
            conv = m.base(z)
            logits = m.out(m.gap(conv).flatten(1))            # dropout is identity in eval
            g, = torch.autograd.grad(logits[0, k], conv)
            w = g.mean(dim=(2, 3), keepdim=True)
            c = F.relu((w * conv).sum(1))[0]
            c = c / (c.max() + 1e-8)
            heat = F.interpolate(c[None, None], size=(self.size, self.size),
                                 mode="bilinear", align_corners=False)[0, 0]
        return heat.detach().numpy()


def load_predictor(framework: str, model_path: str, size: int):
    path = resolve_model_path(model_path)
    fw = framework.lower()
    if fw == "tensorflow":
        predictor = TensorFlowPredictor(path, size)
    elif fw == "pytorch":
        predictor = PyTorchPredictor(path, size)
    else:
        raise ValueError(f'FRAMEWORK must be "tensorflow" or "pytorch", got {framework!r}')
    predictor.predict(np.zeros((size, size, 3), "float32"))   # warm-up so request 1 is fast
    return predictor
