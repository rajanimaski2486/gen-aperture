"""
Image analysis service for extracting subject-aware search cues from PDF images.

Primary path:
- Use a multimodal model to identify concrete objects, scenes, concepts, image text,
  and object-color phrases that can help stock-photo retrieval.

Fallback path:
- If multimodal analysis fails, fall back to a lightweight palette summary so the
  rest of the pipeline still has some image context.
"""
import base64
import json
import logging
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_ANALYSIS_MODEL = "gpt-4o-mini"
_IMAGE_ANALYSIS_MAX_IMAGES = 3
_IMAGE_ANALYSIS_MAX_DIM = 1024
_IMAGE_ANALYSIS_JPEG_QUALITY = 75
_IMAGE_ANALYSIS_MAX_TOKENS = 400

# Lightweight color naming retained for fallback/debug context only.
_COLOR_NAMES = {
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (128, 128, 128): "grey",
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (255, 165, 0): "orange",
    (128, 0, 128): "purple",
    (255, 192, 203): "pink",
    (165, 42, 42): "brown",
    (0, 128, 128): "teal",
    (0, 0, 128): "navy",
    (0, 128, 0): "dark green",
    (192, 192, 192): "silver",
    (255, 215, 0): "gold",
    (64, 224, 208): "turquoise",
    (245, 245, 220): "beige",
    (128, 0, 0): "maroon",
    (255, 127, 80): "coral",
}


def _closest_color_name(r: int, g: int, b: int) -> str:
    min_dist = float("inf")
    name = "unknown"
    for (cr, cg, cb), cname in _COLOR_NAMES.items():
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            name = cname
    return name


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _get_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Dict[str, str]]:
    small = image.copy()
    small.thumbnail((150, 150))
    small = small.convert("RGB")

    quantized = small.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    if not palette:
        return []

    pixel_counts = Counter(quantized.getdata())
    total_pixels = sum(pixel_counts.values()) or 1

    colors = []
    for idx, count in pixel_counts.most_common(num_colors):
        r, g, b = palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
        colors.append(
            {
                "hex": _rgb_to_hex(r, g, b),
                "name": _closest_color_name(r, g, b),
                "percentage": round(count / total_pixels * 100, 1),
            }
        )

    return colors


def _safe_list(values: Any, limit: int = 8) -> List[str]:
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _image_to_data_url(img_dict: Dict[str, Any]) -> str | None:
    data = img_dict.get("data")
    if not data:
        return None
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image.thumbnail((_IMAGE_ANALYSIS_MAX_DIM, _IMAGE_ANALYSIS_MAX_DIM))
        buff = BytesIO()
        image.save(
            buff,
            format="JPEG",
            quality=_IMAGE_ANALYSIS_JPEG_QUALITY,
            optimize=True,
        )
        encoded = base64.b64encode(buff.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        image_format = str(img_dict.get("format") or "jpeg").lower()
        if image_format == "jpg":
            image_format = "jpeg"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:image/{image_format};base64,{encoded}"


def _fallback_image_analysis(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_image: List[Dict[str, Any]] = []
    all_colors: List[Dict[str, str]] = []

    for i, img_dict in enumerate(images):
        try:
            data = img_dict.get("data")
            if not data:
                continue
            pil_image = Image.open(BytesIO(data)).convert("RGB")
            dominant = _get_dominant_colors(pil_image, num_colors=5)
            per_image.append(
                {
                    "index": i + 1,
                    "page": img_dict.get("page"),
                    "width": img_dict.get("width"),
                    "height": img_dict.get("height"),
                    "dominant_colors": dominant,
                }
            )
            all_colors.extend(dominant)
        except Exception as exc:
            logger.warning("Fallback image analysis failed for image %d: %s", i + 1, exc)

    color_weight: Dict[str, float] = {}
    color_hex: Dict[str, str] = {}
    for color in all_colors:
        name = color["name"]
        color_weight[name] = color_weight.get(name, 0) + color["percentage"]
        color_hex.setdefault(name, color["hex"])

    global_palette = sorted(
        [
            {"name": name, "hex": color_hex[name], "weight": round(weight, 1)}
            for name, weight in color_weight.items()
        ],
        key=lambda item: item["weight"],
        reverse=True,
    )[:6]

    color_str = ", ".join(f"{color['name']} ({color['hex']})" for color in global_palette[:5]) or "unknown"
    summary = (
        f"The document contains {len(per_image)} image(s). "
        f"Dominant color palette: {color_str}. "
        "Subject-aware image extraction was unavailable, so search relied on text and palette context only."
    )

    return {
        "image_count": len(per_image),
        "per_image": per_image,
        "global_palette": global_palette,
        "summary": summary,
        "objects": [],
        "scenes": [],
        "concepts": [],
        "text_in_image": [],
        "scene_phrases": [],
        "visual_style_terms": [],
        "object_color_phrases": [],
        "search_terms": [],
        "analysis_source": "palette_fallback",
    }


def analyze_images(
    images: List[Dict[str, Any]],
    api_key: str | None = None,
    model: str = _IMAGE_ANALYSIS_MODEL,
) -> Dict[str, Any]:
    """
    Analyze extracted images and return subject-aware search cues.

    Returns:
        dict with:
            - image_count
            - summary
            - objects
            - scenes
            - concepts
            - text_in_image
            - scene_phrases
            - visual_style_terms
            - object_color_phrases
            - search_terms
            - global_palette
            - analysis_source
    """
    if not images:
        return {
            "image_count": 0,
            "per_image": [],
            "global_palette": [],
            "summary": "No images found in the document.",
            "objects": [],
            "scenes": [],
            "concepts": [],
            "text_in_image": [],
            "scene_phrases": [],
            "visual_style_terms": [],
            "object_color_phrases": [],
            "search_terms": [],
            "analysis_source": "none",
        }

    if not api_key:
        logger.warning("Image analysis: no API key supplied, using palette fallback")
        return _fallback_image_analysis(images)

    ranked_images = sorted(
        images,
        key=lambda item: (item.get("width") or 0) * (item.get("height") or 0),
        reverse=True,
    )
    selected_images = ranked_images[:_IMAGE_ANALYSIS_MAX_IMAGES]
    image_blocks = []
    for img_dict in selected_images:
        data_url = _image_to_data_url(img_dict)
        if data_url:
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "low"},
                }
            )

    if not image_blocks:
        logger.warning("Image analysis: no usable image bytes, using palette fallback")
        return _fallback_image_analysis(images)

    system_prompt = """You analyze uploaded creative-brief images for stock-photo search enrichment.

Your job is to identify subject-aware, retrieval-helpful image cues.

Return JSON only with these keys:
- summary: short 1-2 sentence summary of what the images depict
- objects: list of concrete objects/products visible
- scenes: list of concrete scenes/settings
- text_in_image: list of important readable text/brand/product phrases visible in the images
- visual_style_terms: list of 2-4 short visual descriptors that help find similar stock photos, e.g. "bright vibrant", "sunny tropical", "colorful lifestyle"
- search_terms: list of 4-8 high-confidence subject-aware search phrases helpful for stock-photo retrieval
# Disabled for latency (keep for future re-enable):
# - concepts: list of high-level but retrieval-helpful concepts
# - scene_phrases: list of short scene/setting phrases, e.g. "sunlit beach", "outdoor picnic table"
# - object_color_phrases: list of short phrases attaching color to object/scene

Rules:
- Focus on concrete, visually searchable cues.
- Prefer products, objects, scenes, activities, and branded context.
- Do NOT return generic design-board terms like "brand board", "mood board", "template", "layout", "social media calendar", "SWOT analysis".
- Do NOT return abstract style-only words like "sleek", "modern", "understated", "professional" unless attached to a concrete subject phrase.
- Prefer visual terms that would generalise across many PDFs rather than describing presentation layout.
- Keep phrases short and literal.
- Keep output concise.
"""

    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyze these extracted PDF images and return structured subject-aware cues "
                "for stock-photo search enrichment. Return concise JSON only."
            ),
        }
    ]
    user_content.extend(image_blocks)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=_IMAGE_ANALYSIS_MAX_TOKENS,
        )
        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)
    except Exception as exc:
        logger.warning("Image analysis multimodal call failed, using palette fallback: %s", exc)
        return _fallback_image_analysis(images)

    fallback = _fallback_image_analysis(images)

    objects = _safe_list(parsed.get("objects"), limit=8)
    scenes = _safe_list(parsed.get("scenes"), limit=6)
    concepts = _safe_list(parsed.get("concepts"), limit=6)
    text_in_image = _safe_list(parsed.get("text_in_image"), limit=8)
    scene_phrases = _safe_list(parsed.get("scene_phrases"), limit=6)
    visual_style_terms = _safe_list(parsed.get("visual_style_terms"), limit=4)
    object_color_phrases = _safe_list(parsed.get("object_color_phrases"), limit=6)
    search_terms = _safe_list(parsed.get("search_terms"), limit=8)
    summary = str(parsed.get("summary") or "").strip() or fallback["summary"]

    logger.info(
        "Image analysis complete: %d images, objects=%s, scenes=%s, search_terms=%s",
        len(images),
        objects,
        scenes,
        search_terms,
    )

    return {
        "image_count": len(images),
        "per_image": fallback["per_image"],
        "global_palette": fallback["global_palette"],
        "summary": summary,
        "objects": objects,
        "scenes": scenes,
        "concepts": concepts,
        "text_in_image": text_in_image,
        "scene_phrases": scene_phrases,
        "visual_style_terms": visual_style_terms,
        "object_color_phrases": object_color_phrases,
        "search_terms": search_terms,
        "analysis_source": "multimodal",
    }
