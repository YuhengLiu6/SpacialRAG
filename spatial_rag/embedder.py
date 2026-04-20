# ===== embedder.py =====
import os
import threading
import time
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from spatial_rag.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DINOV2_BATCH_SIZE,
    DINOV2_MODEL_NAME,
    DINOV2_NORMALIZE,
)

try:
    import open_clip  # type: ignore
except Exception:
    open_clip = None  # type: ignore

try:
    import clip as openai_clip  # type: ignore
except Exception:
    openai_clip = None  # type: ignore

try:
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
except Exception:
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore


def _embedder_log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[Embedder][{ts}] {message}", flush=True)


def _preferred_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_openai_clip_model_name(model_name: str) -> str:
    name = str(model_name or "").strip()
    model_map = {
        "ViT-B-16": "ViT-B/16",
        "ViT-B-32": "ViT-B/32",
        "ViT-L-14": "ViT-L/14",
        "ViT-L-14-336": "ViT-L/14@336px",
    }
    return model_map.get(name, name)


class Embedder:
    def __init__(self):
        self.device = _preferred_torch_device()
        self.backend = ""
        _embedder_log(f"loading CLIP model={CLIP_MODEL_NAME} device={self.device}")

        # Optional overrides for offline / pre-downloaded checkpoints.
        pretrained_override = os.environ.get("OPENCLIP_PRETRAINED_PATH")
        pretrained_name_or_path = pretrained_override if pretrained_override else CLIP_PRETRAINED
        cache_dir = os.environ.get("OPENCLIP_CACHE_DIR")
        _embedder_log(
            f"open_clip source pretrained={pretrained_name_or_path} "
            f"cache_dir={cache_dir}"
        )

        try:
            t0 = time.perf_counter()
            if open_clip is not None:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    CLIP_MODEL_NAME,
                    pretrained=pretrained_name_or_path,
                    device=self.device,
                    cache_dir=cache_dir,
                )
                self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
                self.backend = "open_clip"
                _embedder_log(
                    f"open_clip create_model_and_transforms done "
                    f"elapsed_sec={time.perf_counter() - t0:.2f}"
                )
            elif openai_clip is not None:
                openai_model_name = _normalize_openai_clip_model_name(CLIP_MODEL_NAME)
                if pretrained_override:
                    _embedder_log(
                        "OPENCLIP_PRETRAINED_PATH is ignored when using OpenAI CLIP fallback backend"
                    )
                self.model, self.preprocess = openai_clip.load(
                    openai_model_name,
                    device=self.device,
                    download_root=cache_dir,
                )
                self.tokenizer = openai_clip.tokenize
                self.backend = "clip"
                _embedder_log(
                    f"openai clip load done model={openai_model_name} "
                    f"elapsed_sec={time.perf_counter() - t0:.2f}"
                )
            else:
                raise ModuleNotFoundError(
                    "Neither open_clip nor clip is installed in the current Python environment."
                )
        except Exception as exc:
            hint_lines = [
                "Failed to load CLIP model.",
                f"model={CLIP_MODEL_NAME}, pretrained={pretrained_name_or_path}, cache_dir={cache_dir}",
                "This usually means the CLIP package is missing, network/DNS cannot reach model hosting, or no local checkpoint is available.",
                "Fix options:",
                "1) Install open-clip-torch, or ensure the OpenAI clip package is installed.",
                "2) Ensure DNS/network can access model hosting and retry.",
                "3) Pre-download weights and set OPENCLIP_CACHE_DIR to a directory containing cached CLIP weights.",
                "4) If using OpenCLIP, set OPENCLIP_PRETRAINED_PATH to a local checkpoint path.",
            ]
            raise RuntimeError(" ".join(hint_lines)) from exc

        t0 = time.perf_counter()
        self.model.eval()

        # Warmup the model to compile Metal shaders early.
        # This prevents a known driver deadlock between PyTorch MPS and Habitat-Sim's OpenGL context.
        if self.device == 'mps':
            try:
                dummy_img = Image.new("RGB", (224, 224), (0, 0, 0))
                self.embed_image(dummy_img)
                self.embed_text("warmup")
            except Exception as e:
                print(f"Warning: MPS warmup failed: {e}")
        _embedder_log(f"tokenizer/model ready elapsed_sec={time.perf_counter() - t0:.2f}")

    def embed_image(self, image_np):
        """
        Embed an image (numpy array or PIL).
        Args:
            image_np: numpy array (H, W, 3) or PIL Image
        Returns:
            numpy array (D,) normalized embedding
        """
        if not isinstance(image_np, Image.Image):
            image = Image.fromarray(image_np)
        else:
            image = image_np

        t0 = time.perf_counter()
        _embedder_log(f"embed_image start size={image.size}")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        _embedder_log(f"embed_image done elapsed_sec={time.perf_counter() - t0:.2f}")
        return features.cpu().numpy().flatten()

    def embed_text(self, text):
        """
        Embed a text query.
        Args:
            text: str
        Returns:
            numpy array (D,) normalized embedding
        """
        t0 = time.perf_counter()
        _embedder_log(f"embed_text start chars={len(text)}")
        text_tensor = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(text_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        _embedder_log(f"embed_text done elapsed_sec={time.perf_counter() - t0:.2f}")
        return features.cpu().numpy().flatten()


class DINOv2Embedder:
    _MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        model_name: str = DINOV2_MODEL_NAME,
        batch_size: int = DINOV2_BATCH_SIZE,
        normalize: bool = DINOV2_NORMALIZE,
        device: Optional[str] = None,
    ):
        if AutoImageProcessor is None or AutoModel is None:
            raise RuntimeError(
                "transformers is required for DINOv2Embedder. Install `transformers` in the current environment."
            )
        self.model_name = str(model_name or DINOV2_MODEL_NAME).strip() or DINOV2_MODEL_NAME
        self.batch_size = max(1, int(batch_size))
        self.normalize = bool(normalize)
        self.device = str(device or _preferred_torch_device())
        self.model, self.processor = self._load_model_bundle(self.model_name, self.device)

    @classmethod
    def _load_model_bundle(cls, model_name: str, device: str) -> tuple[Any, Any]:
        key = (str(model_name), str(device))
        with cls._CACHE_LOCK:
            cached = cls._MODEL_CACHE.get(key)
            if cached is not None:
                return cached
            t0 = time.perf_counter()
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            model.to(device)
            cls._MODEL_CACHE[key] = (model, processor)
            _embedder_log(
                f"loading DINOv2 model={model_name} device={device} "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )
            return model, processor

    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            arr = np.asarray(image)
            if arr.ndim != 3:
                raise ValueError(f"Expected HWC image ndarray, got shape={arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        if torch.is_tensor(image):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
                tensor = tensor[:3]
                if tensor.dtype.is_floating_point:
                    tensor = torch.clamp(tensor, 0.0, 1.0) * 255.0
                tensor = tensor.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
            elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4}:
                if tensor.shape[-1] == 4:
                    tensor = tensor[..., :3]
                if tensor.dtype.is_floating_point:
                    tensor = torch.clamp(tensor, 0.0, 1.0) * 255.0
                tensor = tensor.to(torch.uint8).numpy()
            else:
                raise ValueError(f"Unsupported tensor image shape={tuple(tensor.shape)}")
            return Image.fromarray(tensor, mode="RGB")
        raise TypeError(f"Unsupported image input type: {type(image).__name__}")

    def _postprocess_features(self, features: torch.Tensor) -> np.ndarray:
        vec = features.detach().cpu().numpy().astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / np.maximum(norms, 1e-12)
        return vec

    def encode_batch(self, images: Sequence[Any]) -> np.ndarray:
        pil_images: List[Image.Image] = []
        for image in list(images or []):
            pil = self._to_pil_image(image)
            if pil.width <= 0 or pil.height <= 0:
                raise ValueError("DINOv2 input image is empty.")
            pil_images.append(pil)
        if not pil_images:
            return np.zeros((0, 0), dtype=np.float32)

        outputs: List[np.ndarray] = []
        for start in range(0, len(pil_images), self.batch_size):
            chunk = pil_images[start:start + self.batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                model_out = self.model(pixel_values=pixel_values)
            cls_features = model_out.last_hidden_state[:, 0, :]
            outputs.append(self._postprocess_features(cls_features))
        return np.vstack(outputs).astype(np.float32)

    def encode_crop(self, image: Any) -> np.ndarray:
        batch = self.encode_batch([image])
        if batch.ndim != 2 or batch.shape[0] != 1:
            raise ValueError(f"Unexpected DINOv2 embedding batch shape={batch.shape}")
        return batch[0]
