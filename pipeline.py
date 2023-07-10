import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Dataset, Input, Output

from typing import List

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['appengine-python-standard']
)
def get_images(directory: str) -> List[str]:
    import os

    # Walk through directory including subdirectories
    matching_files = []
    for root, dirs, files in os.walk(directory.replace("gs://", "/gcs/")):
        for file in files:
            # Check file extension, and add to the list if it is an image
            if file.endswith(".jpg") or file.endswith(".png"):
                matching_files.append(os.path.join(root, file))

    # Return the list of matching files
    return [matching_file.replace("/gcs/", "gs://") for matching_file in matching_files]

@dsl.component(
    base_image='pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime',
    packages_to_install=['transformers', 'torch', 'accelerate', 'Pillow', 'appengine-python-standard']
)
def generate_captions(images: List[str]) -> List[str]:
    import torch

    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    )
    model.to(device)

    captions = []
    for image in images:
        pil_image = Image.open(image.replace("gs://", "/gcs/"))

        inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)

    return captions

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['appengine-python-standard']
)
def write_captions(images: List[str], captions: List[str]):
    for i in range(len(images)):
        image = images[i].replace("gs://", "/gcs/").replace(".jpg", ".txt").replace(".png", ".txt")
        with open(image, "w") as f:
            f.write(captions[i])


@dsl.pipeline(
    name="caption-generator"
)
def caption_generator():
    get_images_task = get_images(
        directory="gs://CHANGE ME/"
    )
    generate_captions_task = generate_captions(
        images=get_images_task.output
    )
    generate_captions_task.set_cpu_request("8")
    generate_captions_task.set_cpu_limit("8")
    generate_captions_task.set_memory_request("32Gi")
    generate_captions_task.set_memory_limit("32Gi")
    generate_captions_task.set_accelerator_limit("2")
    generate_captions_task.set_accelerator_type("NVIDIA_TESLA_T4")
    write_captions(
        images=get_images_task.output,
        captions=generate_captions_task.output
    )


compiler.Compiler().compile(caption_generator, 'pipeline.json')