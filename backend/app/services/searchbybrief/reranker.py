"""
Stage 3: Rerank Candidate Images
"""

import mlx_vlm
import numpy as np
from PIL import Image


# Load the Reranker (8B fits comfortably in 4-bit)
reranker_path = "mlx-community/Qwen3-VL-Reranker-8B-4bit"
r_model, r_processor = mlx_vlm.load(reranker_path)


def run_reranker_node(state: AgentState):
    candidates = state["candidate_pool"]
    query = state["user_request"]
    scored_candidates = []

    # Process in batches for M4 Pro efficiency
    batch_size = 8 
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        images = [Image.open(c["path"]) for c in batch]
        
        # The Reranker expects (Query, Image) pairs
        # In Qwen3-VL-Reranker, we look at the specific 'relevance' logit
        scores = mlx_vlm.compute_scores(
            r_model, 
            r_processor, 
            text=query, 
            images=images
        )
        
        for idx, score in enumerate(scores):
            batch[idx]["rerank_score"] = float(score)
            scored_candidates.append(batch[idx])

    # Sort by score and take the top 500 for the Curator
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    return {"refined_pool": scored_candidates[:500]}