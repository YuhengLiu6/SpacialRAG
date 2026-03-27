import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from query_frame_scorer import FrameQueryScorer, set_ce_loss

class Method3Scorer:
    def __init__(self, embedder, trained_scorer_path=None):
        self.embedder = embedder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dim = self.embedder.model.visual.output_dim
        self.scorer = FrameQueryScorer(d=dim, hidden=256).to(self.device)
        
        # Load weights if a pre-trained scorer path is provided
        if trained_scorer_path and os.path.exists(trained_scorer_path):
            self.scorer.load_state_dict(torch.load(trained_scorer_path, map_location=self.device))
            self.scorer.eval()
            self.is_trained = True
        else:
            self.is_trained = False
            print("WARNING: Loading an untrained Method 3 Frame Scorer. It will yield random results until `.train()` is called.")

    def run(self, query_text, image_paths):
        """
        Runs Method 3 (Frame Scorer MLP) on a new query and set of images.
        """
        self.scorer.eval()
        from PIL import Image
        
        # 1. Embed Query
        q_tensor = self.embedder.tokenizer(query_text).to(self.embedder.device)
        with torch.no_grad():
            q_emb = self.embedder.model.encode_text(q_tensor).cpu() # [1, d]
            
        # 2. Embed Images
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        with torch.no_grad():
            img_tensors = torch.stack([self.embedder.preprocess(img) for img in imgs]).to(self.embedder.device)
            embs = []
            for i in range(0, img_tensors.size(0), 16):
                batch_emb = self.embedder.model.encode_image(img_tensors[i:i+16])
                embs.append(batch_emb.cpu())
            img_embs = torch.cat(embs, dim=0).unsqueeze(0) # [1, N, d]
            
        # 3. Forward pass through FrameQueryScorer
        # Create a valid_mask that is entirely True since there's no padding for a single batch call.
        valid_mask = torch.ones(1, len(image_paths), dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            logits_frames, logit_none = self.scorer(
                img_embs.to(self.device), 
                q_emb.to(self.device), 
                valid_mask=valid_mask
            )
            
            # Combine to get full probability distribution
            logits_all = torch.cat([logits_frames, logit_none], dim=-1)
            probs = F.softmax(logits_all, dim=-1).squeeze(0) # [N + 1]
            
        # 4. Evaluate Prediction
        top1_idx = torch.argmax(probs).item()
        Nmax = len(image_paths)
        
        pred_is_none = (top1_idx == Nmax)
        top1_score = probs[top1_idx].item()
        
        result = {
            "method": "Method 3 (Frame Scorer)",
            "top1_idx": None if pred_is_none else top1_idx,
            "top1_score": top1_score,
            "all_probs": probs.tolist(),
            "best_frame_path": None if pred_is_none else image_paths[top1_idx],
            "is_none": pred_is_none
        }
        
        return result

# For backward compatibility with the old benchmark suite
def evaluate_method3_scorer(dataset, epochs=150, lr=1e-3):
    import torch.optim as optim
    print("\n" + "="*50)
    print("Method 3: Trainable FrameQueryScorer")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_list, B_list, pos_mask_list, valid_mask_list = [], [], [], []
    
    max_N = max(t["num_frames"] for t in dataset)
    dim = dataset[0]["query_embedding"].size(-1)
    
    for task in dataset:
        N = task["num_frames"]
        A_pad = torch.zeros((max_N, dim))
        A_pad[:N, :] = task["image_embeddings"]
        v_mask, p_mask = torch.zeros(max_N, dtype=torch.bool), torch.zeros(max_N, dtype=torch.bool)
        v_mask[:N], p_mask[:N] = True, task["pos_mask"]
        
        A_list.append(A_pad)
        B_list.append(task["query_embedding"])
        pos_mask_list.append(p_mask)
        valid_mask_list.append(v_mask)
        
    A_batch = torch.stack(A_list).to(device)
    B_batch = torch.stack(B_list).to(device)
    pos_mask_batch = torch.stack(pos_mask_list).to(device)
    valid_mask_batch = torch.stack(valid_mask_list).to(device)
    
    model = FrameQueryScorer(d=dim, hidden=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    print("Training the FrameQueryScorer...")
    for ep in range(epochs):
        optimizer.zero_grad()
        logits_frames, logit_none = model(A_batch, B_batch, valid_mask=valid_mask_batch)
        loss = set_ce_loss(logits_frames, logit_none, pos_mask_batch, valid_mask=valid_mask_batch)
        loss.backward()
        optimizer.step()
    
    # Save the optimal weights so we can load it in the actual object class
    save_path = os.path.join(os.path.dirname(__file__), 'method3_trained.pth')
    torch.save(model.state_dict(), save_path)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        logits_frames, logit_none = model(A_batch, B_batch, valid_mask=valid_mask_batch)
        probs = F.softmax(torch.cat([logits_frames, logit_none], dim=-1), dim=-1)
        
        for i, task in enumerate(dataset):
            q_probs = probs[i]
            Nmax = logits_frames.size(1)
            p_mask = pos_mask_batch[i]
            
            top1_idx = torch.argmax(q_probs).item()
            pred_is_none = (top1_idx == Nmax)
            top1_score = q_probs[top1_idx].item()
            gt_is_none = not p_mask.any().item()
            
            if gt_is_none:
                hit = pred_is_none
                pred_str = "None" if pred_is_none else f"Frame {top1_idx}"
            else:
                hit = False if pred_is_none else p_mask[top1_idx].item()
                pred_str = "None" if pred_is_none else f"Frame {top1_idx}"
                    
            results.append({"hit": hit})
            print(f"Task {task['query_id']}: GT_None={gt_is_none} | Pred={pred_str} (Prob={top1_score:.3f}) | Hit={hit}")
            
    acc = sum(r["hit"] for r in results) / len(results)
    print(f"-> Method 3 Overall Accuracy: {acc*100:.1f}%")
    return acc
