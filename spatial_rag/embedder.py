# ===== embedder.py =====
import os
import time

import torch
import open_clip
from PIL import Image
from spatial_rag.config import CLIP_MODEL_NAME, CLIP_PRETRAINED


def _embedder_log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[Embedder][{ts}] {message}", flush=True)


class Embedder:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
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
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=pretrained_name_or_path,
                device=self.device,
                cache_dir=cache_dir,
            )
            _embedder_log(
                f"open_clip create_model_and_transforms done "
                f"elapsed_sec={time.perf_counter() - t0:.2f}"
            )
        except Exception as exc:
            hint_lines = [
                "Failed to load CLIP model.",
                f"model={CLIP_MODEL_NAME}, pretrained={pretrained_name_or_path}, cache_dir={cache_dir}",
                "This usually means network/DNS cannot reach model hosting or no local checkpoint is available.",
                "Fix options:",
                "1) Ensure DNS/network can access huggingface.co and retry.",
                "2) Pre-download weights and set OPENCLIP_PRETRAINED_PATH to the local checkpoint path.",
                "3) Set OPENCLIP_CACHE_DIR to a directory containing cached OpenCLIP weights.",
            ]
            raise RuntimeError(" ".join(hint_lines)) from exc

        t0 = time.perf_counter()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
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
