#!/usr/bin/env python3
import argparse
import base64
import os
import socket
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _print_step(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def _load_project_config_if_requested(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import spatial_rag.config  # noqa: F401
        _print_step("load_project_config", True, "Imported spatial_rag.config")
    except Exception as exc:
        _print_step("load_project_config", False, f"{type(exc).__name__}: {exc}")


def _check_env_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        _print_step("openai_key", False, "OPENAI_API_KEY is missing")
        return False
    _print_step("openai_key", True, f"Found key with prefix: {key[:8]}")
    return True


def _check_dns(host: str = "api.openai.com") -> bool:
    try:
        ip = socket.gethostbyname(host)
        _print_step("dns_lookup", True, f"{host} -> {ip}")
        return True
    except Exception as exc:
        _print_step("dns_lookup", False, f"{type(exc).__name__}: {exc}")
        return False


def _check_https_models(timeout_sec: float = 10.0) -> bool:
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        method="GET",
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            _print_step("https_models", True, f"HTTP {code}")
            return True
    except urllib.error.HTTPError as exc:
        # HTTPError means DNS/TLS/network path is alive; auth may still be wrong.
        _print_step("https_models", True, f"HTTP {exc.code} (network reachable)")
        return True
    except Exception as exc:
        _print_step("https_models", False, f"{type(exc).__name__}: {exc}")
        return False


def _openai_text_call(model: str, timeout_sec: float = 30.0) -> bool:
    try:
        from openai import OpenAI
    except Exception as exc:
        _print_step("import_openai", False, f"{type(exc).__name__}: {exc}")
        return False

    try:
        client = OpenAI(timeout=timeout_sec)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with: ok"}],
            max_tokens=8,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        _print_step("openai_text_call", True, f"Reply: {text!r}")
        return True
    except Exception as exc:
        _print_step("openai_text_call", False, f"{type(exc).__name__}: {exc}")
        return False


def _find_default_image() -> Optional[Path]:
    image_root = Path("spatial_db_obj_strict/images")
    if not image_root.exists():
        return None
    imgs = sorted(image_root.glob("*.jpg"))
    return imgs[0] if imgs else None


def _openai_vision_call(model: str, image_path: Path, timeout_sec: float = 30.0) -> bool:
    try:
        from openai import OpenAI
    except Exception as exc:
        _print_step("import_openai", False, f"{type(exc).__name__}: {exc}")
        return False

    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        client = OpenAI(timeout=timeout_sec)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one short sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
                        },
                    ],
                }
            ],
            max_tokens=64,
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        _print_step("openai_vision_call", True, f"Reply: {text!r}")
        return True
    except Exception as exc:
        _print_step("openai_vision_call", False, f"{type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal OpenAI connectivity check for this project.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to test.")
    parser.add_argument(
        "--load-project-config",
        action="store_true",
        help="Import spatial_rag.config before checks (matches project runtime side effects).",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Also run a vision call using --image or auto-detected local image.",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to image for --vision.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    args = parser.parse_args()

    _load_project_config_if_requested(args.load_project_config)

    ok_key = _check_env_key()
    ok_dns = _check_dns()
    ok_https = _check_https_models(timeout_sec=min(args.timeout, 15.0))

    ok_text = False
    if ok_key and ok_dns:
        ok_text = _openai_text_call(model=args.model, timeout_sec=args.timeout)
    else:
        _print_step("openai_text_call", False, "Skipped because key or DNS check failed")

    ok_vision = True
    if args.vision:
        img = Path(args.image) if args.image else _find_default_image()
        if img is None or not img.exists():
            _print_step("openai_vision_call", False, "No image found; pass --image <path>")
            ok_vision = False
        elif ok_key and ok_dns:
            ok_vision = _openai_vision_call(model=args.model, image_path=img, timeout_sec=args.timeout)
        else:
            _print_step("openai_vision_call", False, "Skipped because key or DNS check failed")
            ok_vision = False

    all_ok = ok_key and ok_dns and ok_https and ok_text and ok_vision
    print()
    print("SUMMARY:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        print("If FAIL persists, run with --load-project-config and check DNS/proxy/firewall.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)

