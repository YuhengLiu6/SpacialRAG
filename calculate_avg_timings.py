import json
import sys
from collections import defaultdict
from pathlib import Path

def main():
    if len(sys.argv) > 1:
        timings_path = Path(sys.argv[1])
    else:
        timings_path = Path("spatial_db_nd/per_image_timings.jsonl")

    if not timings_path.exists():
        print(f"File not found: {timings_path}")
        return

    totals = defaultdict(float)
    counts = defaultdict(int)
    
    with open(timings_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            for key, value in data.items():
                if isinstance(value, (int, float)) and key.endswith("_sec"):
                    totals[key] += float(value)
                    counts[key] += 1
    
    if not totals:
        print("No timing data found.")
        return
        
    averages = {key: (totals[key] / counts[key]) for key in totals}
    
    print(f"--- Average Timings over {counts.get('frame_total_sec', 0)} frames (seconds) ---")
    
    # Define a logical order for the stages based on the pipeline
    ordered_keys = [
        "frame_total_sec",
        "geometry_pipeline_total_sec",
        "selector_sec",
        "dependency_setup_sec",
        "detector_sec",
        "depth_sec",
        "mask_total_sec",
        "angle_geometry_total_sec",
        "crop_vlm_description_total_sec",
        "crop_vlm_description_avg_sec",
        "vlm_fallback_object_parse_sec",
        "fallback_angle_geometry_sec",
        "view_embedding_sec",
        "object_embedding_total_sec"
    ]
    
    # Add any other _sec keys that were found but not in the order
    for key in sorted(totals.keys()):
        if key not in ordered_keys:
            ordered_keys.append(key)
            
    for key in ordered_keys:
        if key in averages:
            print(f"{key:<45}: {averages[key]:.4f}s (from {counts[key]} records)")

if __name__ == "__main__":
    main()
