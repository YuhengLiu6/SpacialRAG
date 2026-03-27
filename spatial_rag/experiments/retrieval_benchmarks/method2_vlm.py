import os
import json
import torch
import torch.nn.functional as F

class Method2VLM:
    def __init__(self, embedder):
        self.embedder = embedder
        self.threshold = 0.60 # Hardcoded none-matching threshold
        self.cache_dir = os.path.join(os.path.dirname(__file__), "vlm_cache")
        self.prompt_version = "dense_caption_distance_v2"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_vlm_caption(self, image_path):
        from openai import OpenAI
        import base64
        import hashlib
        
        # Check cache first to save money and time
        img_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
        cache_key = hashlib.md5(f"{img_hash}|{self.prompt_version}".encode("utf-8")).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return f.read()

        print(f"  [VLM] Calling gpt-4o-mini for {os.path.basename(image_path)}...")
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        detailed_prompt = (
            "You are an expert image analyst. Your task is to extract an exhaustive, detailed inventory of the image. "
            "1. Identify ALL visible objects in the scene, no matter how small.\n"
            "2. For every object, provide a detailed description including its color, material, texture, and state.\n"
            "3. Explicitly describe the exact spatial relationship between all objects (e.g., 'to the left of', 'on top of', 'hanging above').\n"
            "4. For every object, explicitly estimate its approximate physical distance from the camera in meters and mention that estimate in the description.\n"
            "Do not write a generic summary. Be as densely descriptive as possible."
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": detailed_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image and describe every single object and their spatial relationships in extreme detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}", "detail": "low"}}
                        ]
                    }
                ],
                max_tokens=600,
                temperature=0.0
            )
            caption = response.choices[0].message.content.strip()
            
            # Save to cache
            with open(cache_file, 'w') as f:
                f.write(caption)
                
            return caption
        except Exception as e:
            print(f"Error getting VLM caption: {e}")
            return "A view of the room."

    def run(self, query_text, image_paths, save_intermediate_dir=None):
        """
        Runs Method 2 (VLM Proxy) on a new query and set of images.
        Extracts VLM dense descriptions, saves them separately if requested, and computes cosine similarity.
        """
        # 1. Get Text Embedding for User Query
        q_tensor = self.embedder.tokenizer(query_text).to(self.embedder.device)
        with torch.no_grad():
            q_emb = self.embedder.model.encode_text(q_tensor).cpu().squeeze(0) # [d]
            q_emb_norm = F.normalize(q_emb, p=2, dim=-1)

        # 2. Get VLM Captions for all images
        captions = []
        for i, path in enumerate(image_paths):
            cap = self._get_vlm_caption(path)
            captions.append(cap)
            
            # Save intermediate VLM output individually to disk
            if save_intermediate_dir:
                os.makedirs(save_intermediate_dir, exist_ok=True)
                out_path = os.path.join(save_intermediate_dir, f"frame_{i:04d}_caption.txt")
                with open(out_path, 'w') as f:
                    f.write(f"=== Image: {os.path.basename(path)} ===\n")
                    f.write(cap)

        # 3. Get Text Embeddings for the VLM Captions
        cap_embs = []
        for cap in captions:
            with torch.no_grad():
                cap_tensor = self.embedder.tokenizer(cap).to(self.embedder.device)
                c_emb = self.embedder.model.encode_text(cap_tensor).cpu().squeeze(0)
                cap_embs.append(c_emb)
        cap_embs = torch.stack(cap_embs, dim=0) # [N, d]
        cap_embs_norm = F.normalize(cap_embs, p=2, dim=-1)

        # 4. Compute Cosine Similarity (Text-to-Text)
        sim_scores = torch.matmul(cap_embs_norm, q_emb_norm) # [N]
        
        # 5. Predict Top-1
        top1_idx = torch.argmax(sim_scores).item()
        top1_score = sim_scores[top1_idx].item()
        pred_is_none = top1_score < self.threshold
        
        result = {
            "method": "Method 2 (VLM Caption)",
            "top1_idx": None if pred_is_none else top1_idx,
            "top1_score": top1_score,
            "all_scores": sim_scores.tolist(),
            "best_frame_path": None if pred_is_none else image_paths[top1_idx],
            "best_frame_caption": None if pred_is_none else captions[top1_idx],
            "is_none": pred_is_none
        }
        return result

# For backward compatibility with the old benchmark suite
def evaluate_method2_vlm(dataset):
    print("\n" + "="*50)
    print("Method 2: VLM Caption Embeddings (Text-to-Text)")
    print("="*50)
    
    results = []
    for task in dataset:
        q_id = task["query_id"]
        q_emb = task["query_embedding"]
        cap_embs = task["caption_embeddings"]
        pos_mask = task["pos_mask"]
        
        q_emb_norm = F.normalize(q_emb, p=2, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, p=2, dim=-1)
        
        sim_scores = torch.matmul(cap_embs_norm, q_emb_norm)
        
        top1_idx = torch.argmax(sim_scores).item()
        top1_score = sim_scores[top1_idx].item()
        
        THRESHOLD = 0.60
        pred_is_none = top1_score < THRESHOLD
        gt_is_none = not pos_mask.any().item()
        
        if gt_is_none:
            hit = pred_is_none
            pred_str = "None" if pred_is_none else f"Frame {top1_idx}"
        else:
            if pred_is_none:
                hit = False
                pred_str = "None"
            else:
                hit = pos_mask[top1_idx].item()
                pred_str = f"Frame {top1_idx}"
                
        results.append({"hit": hit})
        print(f"Task {q_id}: GT_None={gt_is_none} | Pred={pred_str} (Score={top1_score:.3f}) | Hit={hit}")
        
    acc = sum(r["hit"] for r in results) / len(results)
    print(f"-> Method 2 Overall Accuracy: {acc*100:.1f}%")
    return acc
