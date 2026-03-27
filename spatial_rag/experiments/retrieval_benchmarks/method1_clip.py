import os
import torch
import torch.nn.functional as F

class Method1CLIP:
    def __init__(self, embedder):
        self.embedder = embedder
        self.threshold = 0.25 # Hardcoded none-matching threshold

    def run(self, query_text, image_paths):
        """
        Runs Method 1 (Direct CLIP) on a new query and set of images.
        """
        import sys
        
        # 1. Embed query (Text)
        q_tensor = self.embedder.tokenizer(query_text).to(self.embedder.device)
        with torch.no_grad():
            q_emb = self.embedder.model.encode_text(q_tensor).cpu().squeeze(0) # [d]
            q_emb_norm = F.normalize(q_emb, p=2, dim=-1)

        # 2. Embed images (Vision)
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        
        with torch.no_grad():
            img_tensors = torch.stack([self.embedder.preprocess(img) for img in imgs]).to(self.embedder.device)
            # Process in batches to prevent OOM
            embs = []
            for i in range(0, img_tensors.size(0), 16):
                batch_emb = self.embedder.model.encode_image(img_tensors[i:i+16])
                embs.append(batch_emb.cpu())
            img_embs = torch.cat(embs, dim=0) # [N, d]
            
        img_embs_norm = F.normalize(img_embs, p=2, dim=-1)

        # 3. Compute Cosine Similarity
        sim_scores = torch.matmul(img_embs_norm, q_emb_norm) # [N]
        
        # 4. Predict Top-1
        top1_idx = torch.argmax(sim_scores).item()
        top1_score = sim_scores[top1_idx].item()
        
        pred_is_none = top1_score < self.threshold
        
        result = {
            "method": "Method 1 (CLIP Image)",
            "top1_idx": None if pred_is_none else top1_idx,
            "top1_score": top1_score,
            "all_scores": sim_scores.tolist(),
            "best_frame_path": None if pred_is_none else image_paths[top1_idx],
            "is_none": pred_is_none
        }
        
        return result

# For backward compatibility with the old benchmark suite
def evaluate_method1_clip(dataset):
    print("\n" + "="*50)
    print("Method 1: Direct CLIP Image Embeddings (Cross-modal)")
    print("="*50)
    
    results = []
    for task in dataset:
        q_id = task["query_id"]
        q_emb = task["query_embedding"]
        img_embs = task["image_embeddings"]
        pos_mask = task["pos_mask"]
        
        q_emb_norm = F.normalize(q_emb, p=2, dim=-1)
        img_embs_norm = F.normalize(img_embs, p=2, dim=-1)
        
        sim_scores = torch.matmul(img_embs_norm, q_emb_norm)
        
        top1_idx = torch.argmax(sim_scores).item()
        top1_score = sim_scores[top1_idx].item()
        
        THRESHOLD = 0.25
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
    print(f"-> Method 1 Overall Accuracy: {acc*100:.1f}%")
    return acc
