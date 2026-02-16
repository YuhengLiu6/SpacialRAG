# ===== detector.py =====
from ultralytics import YOLO
from spatial_rag.config import YOLO_MODEL_NAME
import torch

class Detector:
    def __init__(self, model_path=YOLO_MODEL_NAME):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Loading YOLOv8 on {self.device}...")
        self.model = YOLO(model_path) 
        
    def detect(self, image):
        """
        Detect objects in an image.
        Args:
            image: numpy array (H, W, 3) RGB
        Returns:
            list of dicts: {label, bbox, conf}
        """
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
                    "bbox": b,
                    "confidence": float(c)
                })
        
        return detections
