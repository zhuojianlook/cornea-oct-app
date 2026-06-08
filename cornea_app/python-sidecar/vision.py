"""Vision-model calling for paint generation and review.

Ported from the old Rust `call_vision_model` + prompt builders. Three providers:
  - openai   → POST https://api.openai.com/v1/responses
  - local    → POST <base>/chat/completions (OpenAI-compatible, e.g. LM Studio)
  - medgemma → POST <bridge>/review (local_vision/medgemma_bridge.py)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import requests

import settings

TIMEOUT = 240


# ── URL normalisers ────────────────────────────────────────────────────────
def normalized_openai_compatible_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    return trimmed if trimmed.endswith("/chat/completions") else f"{trimmed}/chat/completions"


def normalized_medgemma_bridge_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    try:
        port = urlparse(trimmed).port or 8765
        if port == 1234:
            return "http://127.0.0.1:8765/review"
    except Exception:
        pass
    return trimmed if trimmed.endswith("/review") else f"{trimmed}/review"


# ── Response text extractors ───────────────────────────────────────────────
def collect_response_output_text(value, parts: list[str]) -> None:
    if isinstance(value, dict):
        if value.get("type") == "output_text" and isinstance(value.get("text"), str):
            parts.append(value["text"])
        for child in value.values():
            collect_response_output_text(child, parts)
    elif isinstance(value, list):
        for item in value:
            collect_response_output_text(item, parts)


def collect_chat_completion_text(value) -> str:
    try:
        content = value["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        ).strip()
    return ""


# ── Prompts (verbatim from old Rust) ───────────────────────────────────────
def vision_review_prompt(reviewer_prompt: str) -> str:
    return (
        "You are reviewing OCT cornea segmentation seed-paint previews for research annotation QA, not clinical diagnosis.\n\n"
        "Cyan paint is the cornea seed. Orange paint is the background seed. The target segmentation has only two classes: background and cornea; scars remain part of cornea.\n\n"
        "Judge whether the visible paint is acceptable for Grow from Seeds. Accept only if cyan strokes follow the corneal tissue without crossing into air/background/anterior chamber, orange strokes are clearly outside cornea, and the three-view stack is coherent without obvious clutter or misplaced strokes.\n\n"
        "Return only a JSON object with this exact shape:\n"
        '{"decision":"accept"|"reject","confidence":0.0,"summary":"short text","problems":["short text"],"suggested_feedback_notes":"short text to save if rejected"}\n\n'
        f"Extra reviewer instruction from the user: {reviewer_prompt}"
    )


def vision_paint_prompt(context_metadata: Sequence[dict], feedback, reviewer_prompt: str) -> str:
    image_lines = []
    for i, item in enumerate(context_metadata):
        image_lines.append(
            f"{i + 1}. file={item.get('file_name', 'unknown.png')} "
            f"orientation={item.get('orientation', 'unknown')} "
            f"slice_index={item.get('slice_index', -1)} "
            f"fixed_axis={item.get('fixed_axis', '?')} "
            f"row_axis={item.get('row_axis', '?')} "
            f"column_axis={item.get('column_axis', '?')} "
            f"source={item.get('source_width', 0)}x{item.get('source_height', 0)} "
            f"png={item.get('image_width', 0)}x{item.get('image_height', 0)}"
        )
    feedback_text = json.dumps(feedback, indent=2) if isinstance(feedback, list) else "[]"
    lines = "\n".join(image_lines)
    return (
        "You are the paint-placement agent for research OCT cornea segmentation, not a clinical diagnosis system.\n\n"
        "Your job is to create Grow from Seeds seed paint for two labels only:\n"
        "- cornea: cyan paint, placed inside corneal tissue. Scars stay part of cornea.\n"
        "- background: orange paint, placed outside cornea in air/anterior chamber/other non-corneal areas.\n\n"
        "You are looking at unpainted OCT preview PNGs. Draw curved polyline strokes that follow the visible corneal curvature where possible. Use background strokes to clearly bracket non-corneal areas, but avoid clutter and avoid putting background inside corneal tissue.\n\n"
        "Coordinate rule: return points in PNG pixel coordinates, origin at the top-left of each PNG. Do not return raw IJK coordinates unless you are explicitly using the segments schema. The app will convert PNG pixels to IJK using the metadata below.\n\n"
        f"Image metadata:\n{lines}\n\n"
        f"Prior user/model feedback to incorporate:\n{feedback_text}\n\n"
        "Return only one JSON object with this shape:\n"
        "{\n"
        '  "summary": "short explanation of how you placed the paint",\n'
        '  "confidence": 0.0,\n'
        '  "strokes": [\n'
        '    {"segment":"cornea","image_file":"context_axial_0025.png","radius_voxels":[5,5,2],"points_px":[[x,y],[x,y],[x,y]]},\n'
        '    {"segment":"background","image_file":"context_axial_0025.png","radius_voxels":[6,6,2],"points_px":[[x,y],[x,y],[x,y]]}\n'
        "  ],\n"
        '  "issues": ["short uncertainty notes"]\n'
        "}\n\n"
        "Requirements: include both cornea and background strokes; include axial, coronal, and sagittal views if the anatomy is visible; use polylines/curves rather than isolated dots when possible; keep points inside the PNG dimensions listed above.\n\n"
        f"Extra user instruction: {reviewer_prompt}"
    )


def vision_scar_prompt(context_metadata: Sequence[dict], reviewer_prompt: str) -> str:
    image_lines = "\n".join(
        f"{i + 1}. file={m.get('file_name', '?')} orientation={m.get('orientation', '?')} "
        f"slice_index={m.get('slice_index', -1)} png={m.get('image_width', 0)}x{m.get('image_height', 0)}"
        for i, m in enumerate(context_metadata)
    )
    return (
        "You are the scar-detection agent for research OCT cornea segmentation, not a clinical diagnosis system.\n\n"
        "Scar is an abnormal, often brighter/hazier region WITHIN the corneal tissue. It may be ABSENT — if you "
        "see no clear scar, return an empty strokes list.\n\n"
        "Mark only scar, with label 'scar', staying strictly inside the cornea (do not paint air, anterior "
        "chamber, or normal-looking cornea). Return PNG pixel coordinates (origin top-left); the app converts them.\n\n"
        f"Image metadata:\n{image_lines}\n\n"
        "Return only one JSON object:\n"
        "{\n"
        '  "scar_present": true|false,\n'
        '  "summary": "short text",\n'
        '  "confidence": 0.0,\n'
        '  "strokes": [\n'
        '    {"segment":"scar","image_file":"context_axial_0030.png","radius_voxels":[4,4,2],"points_px":[[x,y],[x,y]]}\n'
        "  ]\n"
        "}\n\n"
        f"Extra user instruction: {reviewer_prompt}"
    )


# ── Provider dispatch ──────────────────────────────────────────────────────
def _openai_key(api_key: str | None) -> str:
    if api_key and api_key.strip():
        return api_key.strip()
    stored = settings.get_settings().get("openai_api_key", "")
    if stored.strip():
        return stored.strip()
    raise ValueError("OpenAI API key missing. Enter one in Settings or set OPENAI_API_KEY.")


def call_vision_model(
    provider_name: str,
    local_base_url: str,
    model_name: str,
    api_key: str | None,
    prompt: str,
    image_paths: Sequence[Path],
    max_new_tokens: int,
) -> dict:
    """Returns {output_text, endpoint, image_files}."""
    responses_content = [{"type": "input_text", "text": prompt}]
    chat_content = [{"type": "text", "text": prompt}]
    medgemma_images = []
    image_files = []
    for idx, path in enumerate(image_paths):
        file_name = path.name
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        medgemma_images.append({"file_name": file_name, "data_url": data_url})
        responses_content.append({"type": "input_text", "text": f"Preview {idx + 1}: {file_name}"})
        responses_content.append({"type": "input_image", "image_url": data_url, "detail": "high"})
        chat_content.append({"type": "text", "text": f"Preview {idx + 1}: {file_name}"})
        chat_content.append({"type": "image_url", "image_url": {"url": data_url}})
        image_files.append(file_name)

    if provider_name == "medgemma":
        endpoint = normalized_medgemma_bridge_url(local_base_url)
        resp = requests.post(
            endpoint,
            json={"prompt": prompt, "images": medgemma_images, "max_new_tokens": max_new_tokens},
            timeout=TIMEOUT,
        )
        if not resp.ok:
            raise ValueError(f"MedGemma bridge error {resp.status_code}: {resp.text}")
        raw = resp.json()
        output_text = (raw.get("output_text") or "").strip()
        if not output_text:
            raise ValueError(f"MedGemma bridge returned no output_text. Response: {resp.text}")
    elif provider_name == "local":
        endpoint = normalized_openai_compatible_url(local_base_url)
        headers = {}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        resp = requests.post(
            endpoint,
            json={"model": model_name, "messages": [{"role": "user", "content": chat_content}],
                  "temperature": 0, "max_tokens": max_new_tokens},
            headers=headers,
            timeout=TIMEOUT,
        )
        if not resp.ok:
            raise ValueError(f"Local vision API error {resp.status_code}: {resp.text}")
        output_text = collect_chat_completion_text(resp.json())
    else:  # openai
        key = _openai_key(api_key)
        endpoint = "https://api.openai.com/v1/responses"
        resp = requests.post(
            endpoint,
            json={"model": model_name, "input": [{"role": "user", "content": responses_content}],
                  "max_output_tokens": max_new_tokens},
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT,
        )
        if not resp.ok:
            raise ValueError(f"OpenAI API error {resp.status_code}: {resp.text}")
        parts: list[str] = []
        collect_response_output_text(resp.json(), parts)
        output_text = "\n".join(parts).strip()

    return {"output_text": output_text, "endpoint": endpoint, "image_files": image_files}
