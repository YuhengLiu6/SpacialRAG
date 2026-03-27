# ===== llm_utils.py =====
import os
from typing import List
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class QueryExpander:
    def __init__(self):
        # Check for API key in environment
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key and OpenAI:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("LLM Client initialized for Query Expansion.")
            except Exception as e:
                print(f"Failed to initialize LLM client: {e}")
        else:
            print("Warning: OPENAI_API_KEY not found or openai module missing. Query expansion will be disabled (passthrough).")

    def expand_query(self, query: str) -> List[str]:
        """
        Expands a vague query into specific visual object targets using an LLM.
        """
        # Fallback if no client
        if not self.client:
            return [query]

        YOLO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
        19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
        24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
        28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
        32: 'sports ball', 33: 'kite', 34: 'baseball bat',
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
        38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
        41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
        53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
        57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv',
        63: 'laptop', 64: 'mouse', 65: 'remote',
        66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster',
        71: 'sink', 72: 'refrigerator',
        73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear',
        78: 'hair drier', 79: 'toothbrush'
        }

        CLASS_LIST = ", ".join(YOLO_CLASSES.values())

        prompt = f"""
        You are a visual search assistant working with a YOLOv8 object detector.

        IMPORTANT:
        You may ONLY output object names from the following list of supported YOLOv8 classes:

        {CLASS_LIST}

        Rules:

        1. If the query refers to a specific object:
        - Map it to the closest class in the allowed list.
        - If multiple relevant classes exist, list them.
        - Do NOT output objects outside the list.

        2. If the query is vague:
        - Translate it into visible objects from the allowed list.

        3. Only output a comma-separated list of class names.
        4. No explanations.
        5. No full sentences.
        6. If no relevant object exists in the list, output the closest reasonable class.

        Examples:

        "find a man" → person
        "portrait" → person
        "canvas" → tv
        "Where can I sleep?" → bed, couch
        "I need to cook" → oven, microwave
        "dog" → dog

        User query: "{query}"
        Output:
        """



#         prompt = f"""
# You are a visual search assistant.

# Your task is to expand a user's query into a list of specific, visible objects that could appear in an image.

# Follow these rules carefully:

# 1. If the query clearly refers to a specific object (e.g., "find a man", "a red chair", "dog"):
#    - Expand it into semantic synonyms or closely related visual terms.
#    - DO NOT introduce unrelated environmental context.
#    - Example:
#      "find a man" → person, human, male, guy
#      "dog" → dog, puppy, canine

# 2. If the query expresses a vague intention (e.g., "Where can I sleep?", "I need to wash my hands"):
#    - Translate it into specific visible objects.
#    - Example:
#      "Where can I sleep?" → bed, couch, sofa, hammock
#      "I need to wash my hands." → sink, washbasin, bathroom

# 3. Only output a comma-separated list of objects.
# 4. Do NOT explain your reasoning.
# 5. Do NOT output full sentences.

# User query: "{query}"
# Output:
# """


        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that lists visible objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50
            )
            content = response.choices[0].message.content.strip()
            # Split by comma and clean up
            objects = [obj.strip() for obj in content.split(',') if obj.strip()]
            
            # Ensure the original query key concept is included if it makes sense, 
            # but usually the LLM covers it. We return the list.
            print(f"Expanded Visual Queries: {objects}")
            return objects
            
        except Exception as e:
            print(f"LLM Query Expansion failed: {e}")
            return [query]
