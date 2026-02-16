# ===== embedder.py =====
import torch
import open_clip
from PIL import Image
from spatial_rag.config import CLIP_MODEL_NAME, CLIP_PRETRAINED

class Embedder:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Loading CLIP ({CLIP_MODEL_NAME}) on {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, 
            pretrained=CLIP_PRETRAINED,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval()
        
        # Get embedding dimension dynamically
        self.output_dim = self.model.visual.output_dim
        print(f"Embedding dimension: {self.output_dim}")

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
            
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy().flatten()

    def embed_text(self, text):
        """
        Embed a text query.
        Args:
            text: str
        Returns:
            numpy array (D,) normalized embedding
        """
        text_tensor = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_text(text_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy().flatten()
