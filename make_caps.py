import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
import json
from PIL import Image
from tqdm import tqdm


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


def parse_args():
    parser = argparse.ArgumentParser(description="Captioning")

    parser.add_argument("--image_dir", default="data/mini_MathVista_grid", help="Image folder that needs captioning and indexing")

    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_caption(img_pth, max_new_tokens, patience=5):
    base64_image = encode_image(img_pth)
    response = None
    attempts = 0

    text_prompt = (
        "Provide a thorough description the image, including key objects, colors, actions, and the overall scene."
        if max_new_tokens >= 100
        else "Provide a brief caption of the image in a descriptive sentence."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=max_new_tokens
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    args = parse_args()


    images = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if img.lower().endswith(("png", "jpg", "jpeg"))]

    annotations = []
    for i, img_pth in tqdm(enumerate(images), total=len(images)):
        description = generate_caption(img_pth, max_new_tokens=150)
        caption = generate_caption(img_pth, max_new_tokens=40)

        annotations.append({
            "path": img_pth,
            "pid": str(i),
            "name": os.path.splitext(os.path.basename(img_pth))[0],
            "caption": caption,
            "description": description
        })

    output = {"annotations": annotations}

    # Save to a JSON file
    output_file = os.path.join(args.image_dir, "caps.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Captions saved to {output_file}")
