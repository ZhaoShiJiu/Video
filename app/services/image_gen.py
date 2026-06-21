"""
AI Image Generation Service

Core logic for:
- Converting video script segments into image generation prompts via LLM
- Calling text-to-image APIs (Dashscope, OpenAI DALL-E, SiliconFlow, Stability AI, ComfyUI)
- Saving generated images to the material directory
- Optional auto-tagging of generated images
"""

import base64
import io
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
from loguru import logger
from openai import OpenAI
from PIL import Image

from app.config import config


# ============================================================
#  Provider-specific implementations
# ============================================================

def _generate_openai_compatible(prompt: str, api_key: str, cfg: dict) -> List[str]:
    """
    Generate images via any OpenAI-compatible image API.

    Uses the OpenAI SDK to call the images/generations endpoint.
    Works with OpenAI DALL-E, SiliconFlow, Dashscope (OpenAI-compat mode),
    and any other provider that implements OpenAI's image API.
    """
    provider = cfg.get("provider", "openai").lower()
    base_url = cfg.get("base_url_override", "").strip()

    # For Dashscope, the base_url_override in config typically points to the
    # native API root (api/v1), not the compatible-mode endpoint. Always use
    # the correct compatible-mode URL for Dashscope to avoid 404 errors.
    if provider == "dashscope":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    elif not base_url:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "siliconflow":
            base_url = "https://api.siliconflow.cn/v1"
        else:
            base_url = "https://api.openai.com/v1"

    model = cfg.get("model", "dall-e-3")
    n = int(cfg.get("images_per_prompt", 1))
    size = cfg.get("image_size", "1792x1024")

    client = OpenAI(api_key=api_key, base_url=base_url)

    images_b64 = []
    for _ in range(n):
        try:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                response_format="b64_json",
                n=1,
            )
            images_b64.append(response.data[0].b64_json)
        except Exception as e:
            logger.error(f"OpenAI-compatible image generation failed: {e}")
            raise

    return images_b64


def _generate_dashscope_native(prompt: str, api_key: str, cfg: dict) -> List[str]:
    """
    Generate images via Dashscope (Alibaba Bailian) native REST API.

    Uses the multimodal-generation endpoint which supports z-image-turbo
    and other Z-Image series models.

    API reference: https://help.aliyun.com/zh/model-studio/z-image
    """
    model = cfg.get("model", "z-image-turbo")
    n = int(cfg.get("images_per_prompt", 1))
    size_str = cfg.get("image_size", "1792x1024")

    # Z-Image uses "W*H" format (e.g. "720*1280")
    size_value = size_str.replace("x", "*")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    images_b64 = []
    for _ in range(n):
        body = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {
                "size": size_value,
                "prompt_extend": False,
            },
        }

        try:
            response = requests.post(url, headers=headers, json=body, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Check for Dashscope error
            if "code" in data and data["code"] != "":
                raise RuntimeError(
                    f"Dashscope API error: code={data.get('code')}, "
                    f"message={data.get('message', 'unknown')}"
                )

            # Z-Image response format:
            # output.choices[0].message.content[] contains
            #   {"image": "<download_url>"} and/or {"text": "<prompt_text>"}
            choices = data.get("output", {}).get("choices", [])
            for choice in choices:
                for item in choice.get("message", {}).get("content", []):
                    if "image" in item:
                        # Download from URL and convert to base64
                        img_resp = requests.get(item["image"], timeout=60)
                        img_resp.raise_for_status()
                        images_b64.append(
                            base64.b64encode(img_resp.content).decode()
                        )

        except Exception as e:
            logger.error(f"Dashscope native image generation failed: {e}")
            raise

    return images_b64


def _generate_siliconflow(prompt: str, api_key: str, cfg: dict) -> List[str]:
    """
    Generate images via SiliconFlow (OpenAI-compatible image API).

    SiliconFlow supports Stable Diffusion models via an OpenAI-compatible
    images/generations endpoint.
    """
    base_url = cfg.get("base_url_override", "").strip() or "https://api.siliconflow.cn/v1"
    model = cfg.get("model", "stabilityai/stable-diffusion-xl-base-1.0")
    n = int(cfg.get("images_per_prompt", 1))
    size = cfg.get("image_size", "1792x1024")

    client = OpenAI(api_key=api_key, base_url=base_url)
    images_b64 = []
    for _ in range(n):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            response_format="b64_json",
            n=1,
        )
        images_b64.append(response.data[0].b64_json)

    return images_b64


def _generate_stability(prompt: str, cfg: dict) -> List[str]:
    """
    Generate images via Stability AI REST API.

    Uses the v1/generation endpoint with the configured engine.
    """
    api_key = cfg.get("stability_api_key", "")
    if not api_key:
        raise ValueError("stability_api_key is required for Stability AI provider")

    engine = cfg.get("stability_engine", "stable-diffusion-xl-1024-v1-0")
    url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"

    size_str = cfg.get("image_size", "1792x1024")
    try:
        w, h = size_str.split("x")
    except ValueError:
        w, h = "1024", "1024"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "text_prompts": [{"text": prompt, "weight": 1.0}],
        "cfg_scale": 7,
        "height": int(h),
        "width": int(w),
        "samples": cfg.get("images_per_prompt", 1),
        "steps": 30,
    }

    response = requests.post(url, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    data = response.json()

    images_b64 = [artifact["base64"] for artifact in data.get("artifacts", [])]
    return images_b64


def _generate_comfyui(prompt: str, cfg: dict) -> List[str]:
    """
    Generate images via local ComfyUI API.

    Sends a prompt to a running ComfyUI instance and retrieves the output.
    """
    base_url = cfg.get("comfyui_base_url", "http://127.0.0.1:8188")
    timeout_val = int(cfg.get("comfyui_timeout", 120))
    size_str = cfg.get("image_size", "1792x1024")
    try:
        w, h = size_str.split("x")
        w, h = int(w), int(h)
    except ValueError:
        w, h = 1024, 1024

    workflow_path = cfg.get("comfyui_workflow_template", "")
    if workflow_path and os.path.isfile(workflow_path):
        with open(workflow_path, "r") as f:
            workflow = json.load(f)
    else:
        workflow = _default_comfyui_workflow(prompt, w, h)

    _inject_prompt_to_workflow(workflow, prompt)

    prompt_resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    prompt_resp.raise_for_status()
    prompt_id = prompt_resp.json()["prompt_id"]

    start = time.time()
    while time.time() - start < timeout_val:
        history_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
        history_resp.raise_for_status()
        history = history_resp.json()
        if prompt_id in history:
            outputs = history[prompt_id]["outputs"]
            images_b64 = []
            for node_id, node_output in outputs.items():
                for img_data in node_output.get("images", []):
                    img_url = (
                        f"{base_url}/view?"
                        f"filename={img_data['filename']}&"
                        f"subfolder={img_data.get('subfolder', '')}&"
                        f"type={img_data.get('type', 'output')}"
                    )
                    img_resp = requests.get(img_url, timeout=30)
                    images_b64.append(base64.b64encode(img_resp.content).decode())
            return images_b64
        time.sleep(1)

    raise TimeoutError(f"ComfyUI generation timed out after {timeout_val}s")


def _inject_prompt_to_workflow(workflow: dict, prompt: str):
    """Inject the positive prompt into a ComfyUI workflow JSON."""
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            widget_values = node.get("_meta", {}).get("title", "")
            if "positive" in str(widget_values).lower() or "正面" in str(widget_values):
                node["inputs"]["text"] = prompt
                return
    # Fallback: find any CLIPTextEncode that looks like a positive prompt input
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            current = node.get("inputs", {}).get("text", "")
            if "negative" not in current.lower() and "负面" not in current:
                node["inputs"]["text"] = prompt
                return


def _default_comfyui_workflow(prompt: str, width: int, height: int) -> dict:
    """Return a minimal SDXL ComfyUI workflow."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0, "steps": 20, "cfg": 7, "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0],
                "negative": ["7", 0], "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "mpt_gen", "images": ["8", 0]},
        },
    }


# ============================================================
#  Image saving utilities
# ============================================================

def _save_b64_images(images_b64: List[str], save_dir: str, output_format: str) -> List[str]:
    """
    Save base64-encoded images to disk.

    Returns a list of absolute file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for b64_data in images_b64:
        img_bytes = base64.b64decode(b64_data)
        filename = f"ai_gen_{uuid.uuid4().hex[:12]}.{output_format}"
        filepath = os.path.join(save_dir, filename)

        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.save(filepath, format=output_format.upper())
        except Exception:
            with open(filepath, "wb") as f:
                f.write(img_bytes)

        saved_paths.append(filepath)
        logger.info(f"Saved generated image: {filepath}")

    return saved_paths


# ============================================================
#  Prompt enrichment
# ============================================================

def _enrich_prompt(prompt: str, cfg: dict) -> str:
    """Append style suffix and negative prompt from config."""
    custom_style = cfg.get("custom_style_prompt", "").strip()
    if custom_style:
        prompt = f"{prompt}, {custom_style}"

    prompt_style = cfg.get("prompt_style", "auto")
    if prompt_style == "anime":
        prompt += ", anime style, vibrant colors, Studio Ghibli inspired, 9:16 composition"
    elif prompt_style == "realistic":
        prompt += ", photorealistic, 8K, cinematic lighting, highly detailed, shallow depth of field"
    elif prompt_style == "illustration":
        prompt += ", digital illustration, clean lines, storybook art style, warm color palette"

    return prompt


# ============================================================
#  API key resolution
# ============================================================

def _resolve_api_key(cfg: dict) -> str:
    """Resolve API key: config override → provider-specific → shared LLM key."""
    override = cfg.get("api_key_override", "").strip()
    if override:
        return override

    provider = cfg.get("provider", "openai").lower()
    if provider == "openai":
        return config.app.get("openai_api_key", "")
    elif provider in ("siliconflow", "dashscope"):
        # Try multiple possible key sources
        siliconflow_key = (
            config.siliconflow.get("api_key", "") if hasattr(config, "siliconflow") else ""
        )
        qwen_key = config.app.get("qwen_api_key", "")
        openai_key = config.app.get("openai_api_key", "")
        return siliconflow_key or qwen_key or openai_key
    elif provider == "stability":
        return cfg.get("stability_api_key", "")
    elif provider == "comfyui":
        return ""  # ComfyUI usually doesn't require auth
    return ""


# ============================================================
#  Auto-tagging generated images
# ============================================================

def _auto_tag_generated_images(image_paths: List[str]):
    """
    Run vision model tagging on newly generated images.

    This is done synchronously because the tags are immediately useful
    for the current video generation task.
    """
    try:
        from app.services import tagging

        material_dir = config.app.get("material_directory", "").strip()
        if not material_dir:
            from app.utils import utils
            material_dir = utils.storage_dir("local_videos", create=True)

        for img_path in image_paths:
            try:
                tagging.tag_single_image(img_path, material_dir)
                logger.info(f"Auto-tagged generated image: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to auto-tag {img_path}: {e}")
    except Exception as e:
        logger.warning(f"Auto-tagging batch failed: {e}")


# ============================================================
#  Single prompt generation dispatch
# ============================================================

def _generate_single_prompt(prompt: str, provider: str, api_key: str, cfg: dict) -> List[str]:
    """Dispatch to the correct provider implementation."""
    provider_lower = provider.lower()

    if provider_lower in ("openai", "siliconflow"):
        # Use OpenAI-compatible API
        return _generate_openai_compatible(prompt, api_key, cfg)
    elif provider_lower == "dashscope":
        # Try native Dashscope API first, fallback to OpenAI-compatible
        try:
            return _generate_dashscope_native(prompt, api_key, cfg)
        except Exception as e:
            logger.warning(
                f"Dashscope native API failed, trying OpenAI-compatible mode: {e}"
            )
            return _generate_openai_compatible(prompt, api_key, cfg)
    elif provider_lower == "stability":
        return _generate_stability(prompt, cfg)
    elif provider_lower == "comfyui":
        return _generate_comfyui(prompt, cfg)
    else:
        # Default: try OpenAI-compatible
        logger.info(f"Unknown provider '{provider}', trying OpenAI-compatible API")
        return _generate_openai_compatible(prompt, api_key, cfg)


# ============================================================
#  Main generation entry point
# ============================================================

def generate_images(
    prompts: List[str],
    save_dir: Optional[str] = None,
    auto_tag: Optional[bool] = None,
) -> List[str]:
    """
    Generate images from a list of text prompts.

    Args:
        prompts: List of image generation prompts (English recommended)
        save_dir: Directory to save generated images
                  (defaults to material_directory/_ai_generated)
        auto_tag: Whether to automatically tag generated images
                  (defaults to config)

    Returns:
        List of absolute paths to saved image files
    """
    cfg = config.image_generation

    if not cfg.get("enabled", False):
        logger.info("Image generation is disabled, skipping")
        return []

    if not prompts:
        logger.warning("No prompts provided for image generation")
        return []

    # Resolve save directory
    if not save_dir:
        material_dir = config.app.get("material_directory", "").strip()
        if not material_dir:
            from app.utils import utils
            material_dir = utils.storage_dir("local_videos", create=True)
        sub_dir = cfg.get("sub_directory", "_ai_generated")
        save_dir = os.path.join(material_dir, sub_dir)

    # Resolve auto_tag
    if auto_tag is None:
        auto_tag = cfg.get("auto_tag", True)

    # Get API credentials
    provider = cfg.get("provider", "openai")
    api_key = _resolve_api_key(cfg)

    if not api_key and provider.lower() not in ("comfyui",):
        logger.warning(
            f"No API key configured for image generation provider '{provider}'. "
            f"Set api_key_override in [image_generation] config."
        )
        return []

    # Generate images
    all_saved_paths = []
    max_images = int(cfg.get("max_images", 8))
    batch_size = int(cfg.get("prompt_batch_size", 3))
    output_format = cfg.get("output_format", "png")

    # Truncate prompts to max_images
    prompts = prompts[:max_images]

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=len(batch_prompts)) as executor:
            futures = {}
            for prompt in batch_prompts:
                full_prompt = _enrich_prompt(prompt, cfg)
                futures[
                    executor.submit(
                        _generate_single_prompt, full_prompt, provider, api_key, cfg
                    )
                ] = prompt

            for future in as_completed(futures):
                prompt = futures[future]
                try:
                    images_b64 = future.result()
                    paths = _save_b64_images(images_b64, save_dir, output_format)
                    all_saved_paths.extend(paths)
                except Exception as e:
                    logger.error(
                        f"Failed to generate image for prompt "
                        f"'{prompt[:80]}...': {e}"
                    )

    logger.info(
        f"Generated {len(all_saved_paths)} images from {len(prompts)} prompts"
    )

    # Auto-tag generated images if enabled
    if auto_tag and all_saved_paths:
        _auto_tag_generated_images(all_saved_paths)

    return all_saved_paths
