from __future__ import annotations

import argparse
import getpass
import json
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe OpenAI-compatible model availability without logging secrets."
    )
    parser.add_argument("--base-url", default="https://sub.qianyueapi.com/v1")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"],
    )
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model IDs before probing them.",
    )
    args = parser.parse_args()
    api_key = getpass.getpass("API key: ").strip()
    if not api_key:
        raise SystemExit("No API key provided.")

    if args.list_models:
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            for model_id in sorted(
                str(item.get("id", "")) for item in result.get("data", [])
            ):
                if model_id:
                    print(model_id)
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"Model listing failed with HTTP {exc.code}.") from None

    for model in args.models:
        started = time.perf_counter()
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Reply with JSON only: {"ok": true}',
                    }
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            args.base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            status = "available" if result.get("choices") else "invalid response"
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
        except Exception as exc:
            status = exc.__class__.__name__
        elapsed = time.perf_counter() - started
        print(f"{model}: {status} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
