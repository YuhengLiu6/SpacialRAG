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

        prompt = f"""
You are a visual search assistant.

Your task is to expand a user's query into a list of specific, visible objects that could appear in an image.

Follow these rules carefully:

1. If the query clearly refers to a specific object (e.g., "find a man", "a red chair", "dog"):
   - Expand it into semantic synonyms or closely related visual terms.
   - DO NOT introduce unrelated environmental context.
   - Example:
     "find a man" → person, human, male, guy
     "dog" → dog, puppy, canine

2. If the query expresses a vague intention (e.g., "Where can I sleep?", "I need to wash my hands"):
   - Translate it into specific visible objects.
   - Example:
     "Where can I sleep?" → bed, couch, sofa, hammock
     "I need to wash my hands." → sink, washbasin, bathroom

3. Only output a comma-separated list of objects.
4. Do NOT explain your reasoning.
5. Do NOT output full sentences.

User query: "{query}"
Output:
"""


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
