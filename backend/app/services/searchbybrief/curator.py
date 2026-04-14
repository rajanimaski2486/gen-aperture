"""
Stage 3: Agentic Curation
"""


import mlx_vlm
from PIL import Image


# Load the model once (35B MoE 4-bit fits easily in M4 Pro 48GB/64GB)
model_path = "mlx-community/Qwen3.5-35B-A3B-4bit"
model, processor = mlx_vlm.load(model_path)


def curator_node(state: AgentState):
    refined_images = state["refined_pool"]
    final_selection = []
    
    # SYSTEM PROMPT for the Curator
    curator_prompt = f"""
    User Request: {state['user_request']}
    Goal: Select the best images from this batch that perfectly match the request.
    Constraint: Ensure diversity in background, lighting, and composition. 
    Review each image and output the IDs of the keepers.
    """

    # For local performance, we process in small visual batches
    batch_size = 5 
    for i in range(0, len(refined_images), batch_size):
        batch = refined_images[i:i + batch_size]
        
        # Prepare images and prompt
        images = [Image.open(img['path']) for img in batch]
        
        # Use MLX-VLM generate
        response = mlx_vlm.generate(
            model, 
            processor, 
            prompt=curator_prompt, 
            images=images,
            max_tokens=1000,
            temp=0.0 # Keep it deterministic for curation
        )
        
        # Logic to parse response and add to final_selection
        # (Assuming Qwen returns a list of IDs or indices)
        selected_ids = parse_ids_from_response(response)
        final_selection.extend([img for img in batch if img['id'] in selected_ids])

    # Check if we met the "100 images" goal
    if len(final_selection) < 100:
        return {
            "final_collection": final_selection,
            "feedback": f"Found {len(final_selection)} images. Need more variety in outdoor settings."
        }
    
    return {"final_collection": final_selection[:100], "feedback": "done"}

