from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from PIL import Image
from ultralytics import YOLO

from spatial_rag.config import (
    DETECTOR_TYPE,
    GROUNDING_DINO_PROMPT,
    YOLO_MODEL_PATH,
    YOLO_WORLD_CLASSES,
    YOLO_WORLD_MODEL_PATH,
)


def _parse_class_names(value: Optional[Union[Iterable[str], str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


class Detector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        detector_type: Optional[str] = None,
        class_names: Optional[Union[Iterable[str], str]] = None,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.detector_type = str(detector_type or DETECTOR_TYPE).strip().upper()
        self.class_names = _parse_class_names(class_names)
        print(f"Initializing Detector: {self.detector_type} on {self.device}...")

        if self.detector_type == "YOLO":
            self.model_path = str(model_path or YOLO_MODEL_PATH)
            print(f"Loading YOLOv8 ({self.model_path})...")
            self.model = YOLO(self.model_path)
        elif self.detector_type == "YOLO_WORLD":
            self.model_path = str(model_path or YOLO_WORLD_MODEL_PATH)
            model_file = Path(self.model_path)
            if not model_file.exists():
                raise FileNotFoundError(
                    f"YOLO-World model not found at {self.model_path}. "
                    "Provide a local yolov8-world checkpoint because this environment is offline."
                )
            print(f"Loading YOLO-World ({self.model_path})...")
            self.model = YOLO(self.model_path)
            if not hasattr(self.model, "set_classes"):
                raise RuntimeError("Loaded model does not support YOLO-World set_classes().")
            names = self.class_names or _parse_class_names(YOLO_WORLD_CLASSES)
            if not names:
                raise ValueError("YOLO_WORLD classes cannot be empty.")
            self.class_names = names
            self.model.set_classes(self.class_names)
            print(f"YOLO-World Classes: {', '.join(self.class_names)}")
        elif self.detector_type == "GROUNDING_DINO":
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "transformers not installed. Run `pip install transformers accelerate`."
                ) from exc

            model_id = "IDEA-Research/grounding-dino-base"
            print(f"Loading Grounding DINO ({model_id})...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.prompt = GROUNDING_DINO_PROMPT.lower()
            if not self.prompt.endswith("."):
                self.prompt += "."
            print(f"Grounding DINO Prompt: {self.prompt}")
        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def set_class_names(self, class_names: Optional[Union[Iterable[str], str]]) -> None:
        names = _parse_class_names(class_names)
        if not names:
            return
        self.class_names = names
        if self.detector_type == "YOLO_WORLD":
            if not hasattr(self.model, "set_classes"):
                raise RuntimeError("Loaded model does not support YOLO-World set_classes().")
            self.model.set_classes(self.class_names)
        elif self.detector_type == "GROUNDING_DINO":
            prompt = ", ".join(self.class_names).lower().strip()
            if prompt and not prompt.endswith("."):
                prompt += "."
            self.prompt = prompt

    def detect(self, image):
        """
        Detect objects in an image.
        Args:
            image: numpy array (H, W, 3) RGB
        Returns:
            list of dicts: {label, bbox, conf}
        """
        if self.detector_type == "YOLO":
            return self._detect_yolo(image)
        if self.detector_type == "YOLO_WORLD":
            return self._detect_yolo_world(image)
        if self.detector_type == "GROUNDING_DINO":
            return self._detect_grounding_dino(image)
        raise ValueError(f"Unknown detector type: {self.detector_type}")

    def _format_yolo_results(self, results) -> List[dict]:
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]
                detections.append(
                    {
                        "label": label,
                        "bbox": bbox,
                        "confidence": float(confidence),
                    }
                )
        return detections

    def _detect_yolo(self, image):
        results = self.model(image, verbose=False, device=self.device)
        return self._format_yolo_results(results)

    def _detect_yolo_world(self, image):
        results = self.model(image, verbose=False, device=self.device)
        return self._format_yolo_results(results)

    def _detect_grounding_dino(self, image_np):
        image_pil = Image.fromarray(image_np)
        inputs = self.processor(images=image_pil, text=self.prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append(
                {
                    "label": label,
                    "bbox": box.cpu().numpy(),
                    "confidence": float(score.cpu().item()),
                }
            )
        return detections
