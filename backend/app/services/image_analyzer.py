"""
Image analysis service for extracting color palettes and mood from PDF images.
Processes extracted image bytes and returns structured analysis for LLM prompts.
"""
import logging
from typing import List, Dict, Any
from io import BytesIO
from collections import Counter

from PIL import Image
import colorsys

logger = logging.getLogger(__name__)

# ── Color-to-name mapping (closest match from a basic palette) ──────────────
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

# ── Color-to-mood mapping ───────────────────────────────────────────────────
_COLOR_MOOD_MAP = {
    "red": ["energetic", "bold", "passionate"],
    "orange": ["warm", "friendly", "enthusiastic"],
    "yellow": ["cheerful", "optimistic", "bright"],
    "gold": ["luxurious", "premium", "sophisticated"],
    "green": ["natural", "fresh", "growth"],
    "dark green": ["natural", "earthy", "stable"],
    "teal": ["calm", "sophisticated", "modern"],
    "turquoise": ["refreshing", "creative", "tranquil"],
    "blue": ["professional", "trustworthy", "calm"],
    "navy": ["corporate", "authoritative", "reliable"],
    "purple": ["creative", "luxurious", "imaginative"],
    "pink": ["playful", "soft", "feminine"],
    "coral": ["warm", "inviting", "trendy"],
    "brown": ["earthy", "rustic", "grounded"],
    "beige": ["neutral", "warm", "understated"],
    "black": ["elegant", "powerful", "sophisticated"],
    "white": ["clean", "minimalist", "pure"],
    "grey": ["neutral", "professional", "modern"],
    "silver": ["sleek", "modern", "industrial"],
    "maroon": ["rich", "serious", "refined"],
}

_SKIP_MOODS = {
    "clean",
    "industrial",
    "minimalist",
    "modern",
    "neutral",
    "pure",
    "sleek",
    "understated",
}


def _closest_color_name(r: int, g: int, b: int) -> str:
    """Find the closest named color for an RGB value."""
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
    """Extract dominant colors from a PIL Image by quantizing."""
    # Resize for speed
    small = image.copy()
    small.thumbnail((150, 150))
    small = small.convert("RGB")

    # Quantize to reduce colors
    quantized = small.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    if not palette:
        return []

    # Count pixel frequency per palette index
    pixel_counts = Counter(quantized.getdata())
    total_pixels = sum(pixel_counts.values())

    colors = []
    for idx, count in pixel_counts.most_common(num_colors):
        r, g, b = palette[idx * 3], palette[idx * 3 + 1], palette[idx * 3 + 2]
        colors.append({
            "hex": _rgb_to_hex(r, g, b),
            "name": _closest_color_name(r, g, b),
            "percentage": round(count / total_pixels * 100, 1),
        })

    return colors


def _infer_mood_from_colors(colors: List[Dict[str, str]]) -> List[str]:
    """Derive mood tags from dominant color names using color psychology."""
    mood_tags: List[str] = []
    for color in colors:
        name = color["name"]
        moods = _COLOR_MOOD_MAP.get(name, [])
        for m in moods:
            if m in _SKIP_MOODS:
                continue
            if m not in mood_tags:
                mood_tags.append(m)
    return mood_tags[:6]  # cap at 6 mood tags


def analyze_images(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a list of extracted images for color palettes and mood.

    Args:
        images: List of image dicts from file_extractor, each with
                'data' (bytes), 'format', 'width', 'height', 'page'.

    Returns:
        dict with:
            - image_count: number of images analyzed
            - per_image: list of per-image analysis dicts
            - global_palette: aggregated dominant colors across all images
            - mood_tags: inferred mood tags from the global palette
            - summary: human-readable summary string for LLM prompts
    """
    if not images:
        return {
            "image_count": 0,
            "per_image": [],
            "global_palette": [],
            "mood_tags": [],
            "summary": "No images found in the document.",
        }

    per_image = []
    all_colors: List[Dict[str, str]] = []

    for i, img_dict in enumerate(images):
        try:
            data = img_dict.get("data")
            if not data:
                continue
            pil_image = Image.open(BytesIO(data)).convert("RGB")
            dominant = _get_dominant_colors(pil_image, num_colors=5)
            per_image.append({
                "index": i + 1,
                "page": img_dict.get("page"),
                "width": img_dict.get("width"),
                "height": img_dict.get("height"),
                "dominant_colors": dominant,
            })
            all_colors.extend(dominant)
        except Exception as e:
            logger.warning(f"Failed to analyze image {i + 1}: {e}")
            continue

    # Aggregate global palette: merge by color name, sum percentages
    color_weight: Dict[str, float] = {}
    color_hex: Dict[str, str] = {}
    for c in all_colors:
        name = c["name"]
        color_weight[name] = color_weight.get(name, 0) + c["percentage"]
        color_hex.setdefault(name, c["hex"])

    global_palette = sorted(
        [{"name": n, "hex": color_hex[n], "weight": round(w, 1)} for n, w in color_weight.items()],
        key=lambda x: x["weight"],
        reverse=True,
    )[:6]

    mood_tags = _infer_mood_from_colors(global_palette)

    # Build human-readable summary
    color_str = ", ".join(f"{c['name']} ({c['hex']})" for c in global_palette[:5])
    mood_str = ", ".join(mood_tags) if mood_tags else "neutral"
    summary = (
        f"The document contains {len(per_image)} image(s). "
        f"Dominant color palette: {color_str}. "
        f"Overall mood/tone: {mood_str}."
    )

    logger.info(f"Image analysis complete: {len(per_image)} images, moods={mood_tags}")

    return {
        "image_count": len(per_image),
        "per_image": per_image,
        "global_palette": global_palette,
        "mood_tags": mood_tags,
        "summary": summary,
    }
