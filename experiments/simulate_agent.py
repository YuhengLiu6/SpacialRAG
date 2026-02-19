
import os
import cv2
import time
import base64
import argparse
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from spatial_rag.config import NUM_STEPS

# Default settings
DEFAULT_STEPS = 30

class TemporalAnalysis(BaseModel):
    found: bool
    relevant_image_ids: List[str]
    reasoning: str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image(image, width, height):
    """Resize image to specific dimensions."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def main():
    parser = argparse.ArgumentParser(description="Simulate agent and analyze with GPT-4o mini")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of steps to simulate")
    parser.add_argument("--queries", type=str, nargs='+', default=["Where did I see a red chair?"], help="List of questions to ask GPT-4o mini")
    parser.add_argument("--width", type=int, default=0, help="Target width for analysis (0 = original)")
    parser.add_argument("--height", type=int, default=0, help="Target height for analysis (0 = original)")
    parser.add_argument("--use_existing", action="store_true", help="Use existing images in output directory instead of simulating new ones")
    parser.add_argument("--output_dir", type=str, default="simulation_images", help="Directory to save/load images")
    parser.add_argument("--save_to_file", type=str, default=None, help="File to save the analysis output (deprecated in favor of results_dir)")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save experiment results (text and images)")
    parser.add_argument("--detail", type=str, default="auto", choices=["auto", "low", "high"], help="Image detail level for OpenAI API")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (e.g., 2 = every other image)")
    parser.add_argument("--query_delay", type=float, default=2.0, help="Delay (seconds) between queries to avoid rate limits")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir

    # Initialize OpenAI Client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return
    
    client = OpenAI(api_key=api_key)

    # Setup output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    saved_images = []

    if args.use_existing:
        print(f"Using existing images from {OUTPUT_DIR}...")
        existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")])
        if len(existing_files) < args.steps:
             print(f"Warning: Found only {len(existing_files)} images, but requested {args.steps} steps.")
        
        # Take the requested number of steps or all available if fewer
        limit = min(args.steps, len(existing_files))
        # Apply downsampling
        saved_images = [os.path.join(OUTPUT_DIR, f) for f in existing_files[:limit]][::args.downsample]
    else:
        # Initialize Explorer
        print("Initializing Explorer...")
        try:
            from spatial_rag.explorer import Explorer
            explorer = Explorer()
            
            print(f"Starting simulation for {args.steps} steps...")
            
            for step in tqdm(range(args.steps)):
                # Move agent
                rgb_image, position, rotation = explorer.step_random()
                
                # Save original image (high quality) for reference
                timestamp = int(time.time() * 1000)
                img_filename = f"sim_step_{step:03d}_{timestamp}.jpg"
                img_path = os.path.join(OUTPUT_DIR, img_filename)
                
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, bgr_image)
                
                # Handle resizing for API if requested
                api_image_path = img_path
                if args.width > 0 and args.height > 0:
                    resized = resize_image(bgr_image, args.width, args.height)
                    resized_filename = f"sim_step_{step:03d}_{timestamp}_{args.width}x{args.height}.jpg"
                    resized_path = os.path.join(OUTPUT_DIR, resized_filename)
                    cv2.imwrite(resized_path, resized) # Save resized version
                    api_image_path = resized_path

                saved_images.append(api_image_path)
            
            # Apply downsampling after collection if simulating
            saved_images = saved_images[::args.downsample]

        except Exception as e:
            print(f"Simulation interrupted or failed: {e}")
            if not saved_images:
                return
        finally:
            if 'explorer' in locals():
                explorer.close()

    if not saved_images:
        print("No images available for analysis.")
        return

    # Handle Results Directory Logic
    result_images = []
    if args.results_dir:
        import shutil
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        
        # Create images subdir
        imgs_dir = os.path.join(args.results_dir, "images")
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir)
            
        print(f"Copying {len(saved_images)} images to {imgs_dir}...")
        for img_path in saved_images:
            basename = os.path.basename(img_path)
            dest_path = os.path.join(imgs_dir, basename)
            shutil.copy2(img_path, dest_path)
            result_images.append(dest_path)
    else:
        result_images = saved_images

    print(f"\nCaptured/Loaded {len(saved_images)} images (Downsample: {args.downsample}). Preparing analysis requests...")

    # Iterate through all queries
    for i, query in enumerate(args.queries):
        if i > 0 and args.query_delay > 0:
            print(f"Waiting {args.query_delay}s before next query...")
            time.sleep(args.query_delay)

        print(f"\n--- Processing Query: '{query}' ---")
        
        # Prepare content with explicit image IDs
        content = [
            {"type": "text", "text": f"Analyze this sequence of {len(saved_images)} images. Answer the question: {query}\n\n"
                                     f"Output MUST be a valid JSON object matching this schema:\n"
                                     f"{json.dumps(TemporalAnalysis.model_json_schema(), indent=2)}\n\n"
                                     f"IMPORTANT: Refer to images ONLY by their explicit ID provided below (e.g., 'step_0'). "
                                     f"If the object is not found, set 'found' to false and 'relevant_image_ids' to empty list."}
        ]

        for idx, img_path in enumerate(saved_images):
            base64_image = encode_image(img_path)
            # Add label before image
            content.append({"type": "text", "text": f"Image ID: step_{idx}"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": args.detail 
                }
            })

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Sending request to GPT-4o mini (Attempt {attempt+1}/{max_retries})...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise video analysis agent. You only output valid JSON. You never hallucinate object presence."
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                result_content = response.choices[0].message.content
                print("\nGPT-4o Mini Analysis (JSON):")
                print(result_content)
                print("----------------------------")
                
                # Consolidate saving logic
                save_path = None
                if args.results_dir:
                    save_path = os.path.join(args.results_dir, "analysis.jsonl") # Changed to jsonl for structured data
                elif args.save_to_file:
                    save_path = args.save_to_file

                if save_path:
                    # Parse to verify it matches schema (optional but good for validation)
                    try:
                        parsed_json = json.loads(result_content)
                        # Enrich with query info
                        output_record = {"query": query, "response": parsed_json}
                        
                        with open(save_path, "a") as f:
                            f.write(json.dumps(output_record) + "\n")
                        print(f"Output saved to {save_path}")
                    except json.JSONDecodeError:
                        print(f"Error: Failed to parse JSON response. Raw output saved.")
                        with open(save_path + ".raw", "a") as f:
                             f.write(f"Query: {query}\n{result_content}\n---\n")

                break # Success, exit retry loop

            except Exception as e:
                print(f"API Request Failed: {e}")
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    print(f"Aborting query '{query}' due to error.")
                    break

if __name__ == "__main__":
    main()
