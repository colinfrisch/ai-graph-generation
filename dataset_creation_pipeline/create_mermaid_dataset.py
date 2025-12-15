import base64
import io
import dotenv
import os

from datasets import load_from_disk
from datasets import Dataset
from litellm import completion


dotenv.load_dotenv()


def llm_request(image_base64, prompt, model="openai/gpt-5.2"):
    """Request Anthropic with an image and prompt"""
    resp = completion(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }]
    )
    
    response_text = resp.choices[0].message.content
    return response_text


def image_to_mermaid_code(image, prompt, model="openai/gpt-5.2"):
    """Convert an image to a Mermaid code"""

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.standard_b64encode(buffer.read()).decode("utf-8")

    # Request the LLM to convert the image to a Mermaid code
    response = llm_request(image_base64, prompt, model)
    
    return response




def main():

    LLM_MODEL = "openai/gpt-5.2"
    PROMPT = "Provide The code for this diagram in Mermaid format. Only the code, no other text. Current caption: {caption}"


    provider = LLM_MODEL.split("/")[0].upper()
    os.environ[f"{provider}_API_KEY"] = os.getenv(f"{provider}_API_KEY")


    new_dataset = []
    diagrams_with_captions = load_from_disk("datasets/diagrams_with_captions")

    # Handle both DatasetDict (with splits) and Dataset (single split)
    if hasattr(diagrams_with_captions, "keys") and "train" in diagrams_with_captions.keys():
        dataset_split = diagrams_with_captions["train"]
    else:
        dataset_split = diagrams_with_captions

    print(f"Dataset columns: {dataset_split.column_names}")
    print(f"First sample keys: {dataset_split[0].keys()}")

    for sample in dataset_split.select(range(min(3, len(dataset_split)))):
        try:
            image = sample["image"]
            caption = sample["text"]
            prompt = PROMPT.format(caption=caption)
            response = image_to_mermaid_code(image, prompt, LLM_MODEL)
            
            print(response)
            new_dataset.append({
                "code": response,
                "caption": caption,
            })
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    if new_dataset:
        diagrams_with_mermaid_codes = Dataset.from_list(new_dataset)
        os.makedirs("datasets", exist_ok=True)
        diagrams_with_mermaid_codes.save_to_disk("datasets/diagrams_with_mermaid_codes")
        print(f"Saved {len(new_dataset)} diagrams with mermaid codes to datasets/diagrams_with_mermaid_codes")
    else:
        print("No data to save. Check for errors above.")

if __name__ == "__main__":
    main()
