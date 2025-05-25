import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)


def fetch_pretrained_model(model_class, model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}..."
                )
            else:
                raise


def get_diffusion_pipelines():
    """
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    Downloads only - no GPU operations during build.
    """
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
        "device_map": None,  # Don't map to any device during download
    }

    print("Downloading SDXL base pipeline...")
    pipe = fetch_pretrained_model(
        StableDiffusionXLPipeline,
        "stabilityai/stable-diffusion-xl-base-1.0",
        **common_args,
    )
    
    print("Downloading VAE...")
    vae = fetch_pretrained_model(
        AutoencoderKL, 
        "madebyollin/sdxl-vae-fp16-fix", 
        **{"torch_dtype": torch.float16}
    )
    
    print("Downloading SDXL refiner pipeline...")
    refiner = fetch_pretrained_model(
        StableDiffusionXLImg2ImgPipeline,
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        **common_args,
    )

    print("All models downloaded successfully!")
    return pipe, refiner, vae


if __name__ == "__main__":
    get_diffusion_pipelines()
