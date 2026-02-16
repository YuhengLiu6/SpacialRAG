# ===== detector.py =====
from ultralytics import YOLO
from spatial_rag.config import YOLO_MODEL_PATH, DETECTOR_TYPE, GROUNDING_DINO_PROMPT
import torch
from PIL import Image
import numpy as np

class Detector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        self.detector_type = DETECTOR_TYPE
        print(f"Initializing Detector: {self.detector_type} on {self.device}...")

        if self.detector_type == "YOLO":
            print(f"Loading YOLOv8 ({model_path})...")
            self.model = YOLO(model_path)
            
        elif self.detector_type == "GROUNDING_DINO":
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            except ImportError:
                print("Error: transformers not installed. Run `pip install transformers accelerate`.")
                exit(1)
                
            model_id = "IDEA-Research/grounding-dino-base"
            print(f"Loading Grounding DINO ({model_id})...")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            # Pre-process prompt
            # Grounding DINO expects lower case, dot separated or specific format?
            # Usually strict: "text . text ."
            self.prompt = GROUNDING_DINO_PROMPT.lower()
            if not self.prompt.endswith("."):
                self.prompt += "."
            print(f"Grounding DINO Prompt: {self.prompt}")

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
        elif self.detector_type == "GROUNDING_DINO":
            return self._detect_grounding_dino(image)
        else:
            print(f"Unknown detector type: {self.detector_type}")
            return []

    def _detect_yolo(self, image):
        results = self.model(image, verbose=False, device=self.device)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                c = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = self.model.names[cls]
                
                detections.append({
                    "label": label,
                    "bbox": b, # [x1, y1, x2, y2]
                    "confidence": float(c)
                })
        return detections

    def _detect_grounding_dino(self, image_np):
        # Convert numpy (RGB) to PIL
        image_pil = Image.fromarray(image_np)
        
        inputs = self.processor(images=image_pil, text=self.prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process
        target_sizes = torch.tensor([image_pil.size[::-1]]) # (H, W)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35, # Sensitivity (renamed from box_threshold)
            text_threshold=0.25,
            target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
             # box is [x1, y1, x2, y2]
             b = box.cpu().numpy()
             c = score.cpu().item()
             l = label # label is string
             
             detections.append({
                 "label": l,
                 "bbox": b,
                 "confidence": float(c)
             })
             
        return detections
