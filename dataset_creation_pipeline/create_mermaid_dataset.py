import asyncio
import base64
import io
import dotenv
import os

from datasets import load_from_disk
from datasets import Dataset
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio


dotenv.load_dotenv()

# Limit concurrent requests to avoid rate limiting
MAX_CONCURRENT_REQUESTS = 50


async def llm_request(image_base64, prompt, model="openai/gpt-5.2", semaphore=None):
    """Request LLM with an image and prompt"""
    if semaphore:
        async with semaphore:
            return await _do_llm_request(image_base64, prompt, model)
    return await _do_llm_request(image_base64, prompt, model)


async def _do_llm_request(image_base64, prompt, model):
    """Actual LLM request logic"""
    resp = await acompletion(
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


async def image_to_mermaid_code(image, prompt, model="openai/gpt-5.2", semaphore=None):
    """Convert an image to a Mermaid code"""

    # Convert CMYK or other modes to RGB (PNG doesn't support CMYK)
    if image.mode in ("CMYK", "P", "LA", "PA"):
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        pass  # RGBA is fine for PNG
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.standard_b64encode(buffer.read()).decode("utf-8")

    # Request the LLM to convert the image to a Mermaid code
    response = await llm_request(image_base64, prompt, model, semaphore)
    
    return response


async def process_sample(sample, prompt_template, model, semaphore):
    """Process a single sample and return the result or None on error"""
    try:
        image = sample["image"]
        caption = sample["text"]
        prompt = prompt_template.format(caption=caption)
        response = await image_to_mermaid_code(image, prompt, model, semaphore)
        
        print(response)
        return {
            "code": response,
            "caption": caption,
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


async def main():

    LLM_MODEL = "openai/gpt-5.2"
    PROMPT = "Provide The code for this diagram in Mermaid format. Only the code, no other text. Current caption: {caption}"

    provider = LLM_MODEL.split("/")[0].upper()
    api_key = os.getenv(f"{provider}_API_KEY")
    if api_key:
        os.environ[f"{provider}_API_KEY"] = api_key

    diagrams_with_captions = load_from_disk("datasets/diagrams_with_captions")

    # Handle both DatasetDict (with splits) and Dataset (single split)
    if hasattr(diagrams_with_captions, "keys") and "train" in diagrams_with_captions.keys():
        dataset_split = diagrams_with_captions["train"]
    else:
        dataset_split = diagrams_with_captions

    print(f"Dataset columns: {dataset_split.column_names}")
    print(f"First sample keys: {dataset_split[0].keys()}")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Get samples to process
    num_samples = len(dataset_split)
    samples = [dataset_split[i] for i in range(num_samples)]

    # Process all samples concurrently
    print(f"Processing {num_samples} samples concurrently...")
    tasks = [process_sample(sample, PROMPT, LLM_MODEL, semaphore) for sample in samples]
    results = await tqdm_asyncio.gather(*tasks, desc="Creating dataset")

    # Filter out None results (errors)
    new_dataset = [r for r in results if r is not None]

    if new_dataset:
        diagrams_with_mermaid_codes = Dataset.from_list(new_dataset)
        os.makedirs("datasets", exist_ok=True)
        diagrams_with_mermaid_codes.save_to_disk("datasets/diagrams_with_mermaid_codes")
        print(f"Saved {len(new_dataset)} diagrams with mermaid codes to datasets/diagrams_with_mermaid_codes")
    else:
        print("No data to save. Check for errors above.")


if __name__ == "__main__":
    asyncio.run(main())
