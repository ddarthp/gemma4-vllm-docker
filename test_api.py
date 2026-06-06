#!/usr/bin/env python3
"""Prueba rapida del endpoint Gemma 4 12B (OpenAI-compatible).
Uso:  python test_api.py            (texto)
      python test_api.py --image URL
      python test_api.py --audio RUTA.wav
"""
import argparse, base64, os
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000/v1"))
ap.add_argument("--model", default=os.environ.get("MODEL", "gemma-4-12b"))
ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
ap.add_argument("--image")
ap.add_argument("--audio")
ap.add_argument("--prompt", default="Explica en una frase que es un grafo dirigido.")
args = ap.parse_args()

client = OpenAI(base_url=args.base_url, api_key=args.api_key)

content = [{"type": "text", "text": args.prompt}]
if args.image:
    content.insert(0, {"type": "image_url", "image_url": {"url": args.image}})
if args.audio:
    with open(args.audio, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    content.insert(0, {"type": "input_audio",
                       "input_audio": {"data": b64, "format": "wav"}})

resp = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": content}],
    max_tokens=512,
    temperature=1.0,      # recomendado por el model card
    top_p=0.95,
    extra_body={"top_k": 64},
)
print(resp.choices[0].message.content)
