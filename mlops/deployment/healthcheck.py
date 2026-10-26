#!/usr/bin/env python3
"""
MedVision-AI — Container health probe.

Used by:
  * Dockerfile   HEALTHCHECK
  * docker-compose.yml healthcheck
  * Kubernetes   livenessProbe / readinessProbe / startupProbe

Exit codes:
    0 — service is healthy and ready to receive traffic
    1 — service is unhealthy (HTTP error, missing dependency, or exception)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("APP_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("APP_PORT", "8000"))
HEALTH_PATH = os.environ.get("HEALTH_PATH", "/health")
TIMEOUT_SECONDS = float(os.environ.get("HEALTH_TIMEOUT", "5"))


def _normalize_host(host: str) -> str:
    """`0.0.0.0` is only a bind address; probes must hit loopback."""
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def check_health() -> int:
    host = _normalize_host(DEFAULT_HOST)
    url = f"http://{host}:{DEFAULT_PORT}{HEALTH_PATH}"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        print(f"[healthcheck] HTTPError {exc.code} from {url}: {exc.reason}")
        return 1
    except urllib.error.URLError as exc:
        print(f"[healthcheck] URLError reaching {url}: {exc.reason}")
        return 1
    except Exception as exc:  # pragma: no cover — defensive
        print(f"[healthcheck] Unexpected error: {exc!r}")
        return 1

    if status != 200:
        print(f"[healthcheck] Non-200 status {status} from {url}")
        return 1

    # Optionally inspect the JSON payload for a richer status field.
    try:
        payload = json.loads(body)
        state = (payload.get("status") or payload.get("state") or "").lower()
        if state and state not in ("ok", "healthy", "ready", "up"):
            print(f"[healthcheck] Service reports status={state!r}: {body[:200]}")
            return 1
    except json.JSONDecodeError:
        # Plain-text 200 is acceptable for liveness probes.
        pass

    print(f"[healthcheck] OK — {url} reachable (HTTP {status})")
    return 0


if __name__ == "__main__":
    sys.exit(check_health())
