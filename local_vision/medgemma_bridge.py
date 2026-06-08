#!/usr/bin/env python3
"""Local MedGemma vision bridge for OCT segmentation paint planning and review."""

import argparse
import base64
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


PIPE = None
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(description="Serve MedGemma image-text requests locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_pipe():
    global PIPE
    if PIPE is not None:
        return PIPE
    import torch
    from transformers import pipeline

    dtype = torch.bfloat16 if ARGS.device == "cuda" and torch.cuda.is_available() else torch.float32
    device = ARGS.device if ARGS.device == "cuda" and torch.cuda.is_available() else -1
    PIPE = pipeline(
        "image-text-to-text",
        model=ARGS.model,
        torch_dtype=dtype,
        device=device,
    )
    return PIPE


def image_from_data_url(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def generated_text_to_string(output):
    generated = output[0].get("generated_text", output[0]) if isinstance(output, list) else output
    if isinstance(generated, list) and generated:
        content = generated[-1].get("content", generated[-1])
    elif isinstance(generated, dict):
        content = generated.get("content", generated)
    else:
        content = generated
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} - {fmt % args}")

    def send_json(self, status, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path in ("/healthz", "/readyz"):
            self.send_json(200, {"ok": True, "model": ARGS.model})
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/review":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = payload.get("prompt") or ""
            images = [
                image_from_data_url(item["data_url"])
                for item in payload.get("images", [])
                if item.get("data_url")
            ]
            if not images:
                self.send_json(400, {"error": "No images supplied"})
                return
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image", "image": image} for image in images)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert medical image segmentation assistant."}],
                },
                {"role": "user", "content": content},
            ]
            pipe = load_pipe()
            output = pipe(
                text=messages,
                max_new_tokens=int(payload.get("max_new_tokens") or ARGS.max_new_tokens),
                do_sample=False,
            )
            self.send_json(
                200,
                {
                    "model": ARGS.model,
                    "image_count": len(images),
                    "output_text": generated_text_to_string(output),
                },
            )
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"error": str(exc)})


def main():
    global ARGS
    ARGS = parse_args()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"MedGemma bridge listening on http://{ARGS.host}:{ARGS.port}/review")
    print(f"Model will load lazily on first request: {ARGS.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
