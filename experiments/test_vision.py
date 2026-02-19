
import os
import base64
import base64
import sys
# Add current directory to path to allow importing spatial_rag
sys.path.append(os.getcwd())
from spatial_rag import config
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API Key found")
        return

    client = OpenAI(api_key=api_key)
    
    # Use one of the generated images
    img_path = "simulation_images/sim_step_000_1771538709659.jpg"
    # Fallback to finding any jpg in the dir if that specific one doesnt exist (it should)
    if not os.path.exists(img_path):
        import glob
        imgs = glob.glob("simulation_images/*.jpg")
        if imgs:
            img_path = imgs[0]
        else:
            print("No images found")
            return

    print(f"Testing vision with {img_path}")
    base64_image = encode_image(img_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
