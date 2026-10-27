"""
MedVision-AI — Application health checks.

Distinguishes between *liveness* (the process is up) and *readiness*
(the process is up AND its dependencies are reachable, so it can serve
traffic). This separation matches Kubernetes' livenessProbe / readinessProbe
semantics.

Usage as a CLI (exit 0 = healthy, exit 1 = unhealthy):

    python -m mlops.monitoring.health_check liveness
    python -m mlops.monitoring.health_check readiness

Usage as a module:

    from mlops.monitoring.health_check import check_liveness, check_readiness
    status = check_readiness()   # -> HealthStatus
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOST = os.environ.get("APP_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("APP_PORT", "8000"))
LIVENESS_PATH = os.environ.get("HEALTH_LIVENESS_PATH", "/health")
READINESS_PATH = os.environ.get("HEALTH_READINESS_PATH", "/ready")
TIMEOUT_SECONDS = float(os.environ.get("HEALTH_TIMEOUT", "5"))

# Comma-separated list of "kind:target" dependency descriptors, e.g.
#   "tcp:redis:6379,tcp:postgres:5432"
DEPENDENCIES_ENV = os.environ.get("HEALTH_DEPENDENCIES", "")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DependencyResult:
    name: str
    kind: str
    target: str
    healthy: bool
    latency_ms: float
    error: str | None = None


@dataclass
class HealthStatus:
    status: str            # "ok" | "degraded" | "unhealthy"
    healthy: bool
    component: str         # "liveness" | "readiness"
    timestamp: float
    checks: list[DependencyResult] = field(default_factory=list)
    http_status: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_host(host: str) -> str:
    """`0.0.0.0` is a bind address; probes must hit loopback."""
    if host in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def _http_get(url: str, timeout: float) -> tuple[int | None, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body
    except (urllib.error.URLError, socket.timeout) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _check_tcp(name: str, target: str, timeout: float = 2.0) -> DependencyResult:
    """Open a TCP connection to verify a backing service is reachable."""
    # Accept host:port, or redis://host:port/0
    parsed = urllib.parse.urlparse(target)
    if parsed.hostname:
        host, port = parsed.hostname, parsed.port or 6379
    elif ":" in target:
        host, _, port_str = target.rpartition(":")
        try:
            port = int(port_str)
        except ValueError:
            return DependencyResult(name, "tcp", target, False, 0.0,
                                    f"invalid port in target: {target!r}")
    else:
        return DependencyResult(name, "tcp", target, False, 0.0,
                                f"unparseable target: {target!r}")

    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency_ms = (time.perf_counter() - start) * 1000.0
            return DependencyResult(name, "tcp", target, True, latency_ms)
    except OSError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return DependencyResult(name, "tcp", target, False, latency_ms,
                                 f"{type(exc).__name__}: {exc}")


def _parse_dependencies(env_value: str) -> list[tuple[str, str, str]]:
    """Parse `kind:target` descriptors. target may itself contain `:port`."""
    deps: list[tuple[str, str, str]] = []
    for raw in env_value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        kind, _, rest = raw.partition(":")
        kind, rest = kind.strip().lower(), rest.strip()
        if not kind or not rest:
            continue
        # name defaults to the target host
        name = rest.split(":", 1)[0].split("/", 1)[-1]
        deps.append((name, kind, rest))
    return deps


def _probe_dependencies(deps: Iterable[tuple[str, str, str]]) -> list[DependencyResult]:
    results: list[DependencyResult] = []
    for name, kind, target in deps:
        if kind == "tcp":
            results.append(_check_tcp(name, target))
        else:
            results.append(DependencyResult(
                name=name, kind=kind, target=target,
                healthy=False, latency_ms=0.0,
                error=f"unsupported dependency kind: {kind!r}",
            ))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_liveness(
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    timeout: float | None = None,
) -> HealthStatus:
    """Liveness probe — is the process up and responding at all?"""
    host = _normalize_host(host or DEFAULT_HOST)
    port = port or DEFAULT_PORT
    path = path or LIVENESS_PATH
    timeout = timeout or TIMEOUT_SECONDS

    url = f"http://{host}:{port}{path}"
    ts = time.time()
    status, body = _http_get(url, timeout)

    if status == 200:
        return HealthStatus(status="ok", healthy=True,
                             component="liveness", timestamp=ts,
                             http_status=status)

    return HealthStatus(
        status="unhealthy", healthy=False,
        component="liveness", timestamp=ts,
        http_status=status,
        error=body or f"liveness probe failed (HTTP {status})",
    )


def check_readiness(
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    timeout: float | None = None,
    dependencies: Iterable[tuple[str, str, str]] | None = None,
) -> HealthStatus:
    """Readiness probe — liveness OK + all backing dependencies reachable."""
    host = _normalize_host(host or DEFAULT_HOST)
    port = port or DEFAULT_PORT
    path = path or READINESS_PATH
    timeout = timeout or TIMEOUT_SECONDS

    # First: liveness must pass.
    live = check_liveness(host, port, path, timeout)
    if not live.healthy:
        # Carry over the liveness failure but tag it as readiness.
        live.component = "readiness"
        return live

    # Then: probe dependencies (default to env-configured list).
    deps = list(dependencies) if dependencies is not None else _parse_dependencies(DEPENDENCIES_ENV)
    dep_results = _probe_dependencies(deps)

    all_healthy = all(r.healthy for r in dep_results)
    overall_status = "ok" if all_healthy else "degraded"

    return HealthStatus(
        status=overall_status,
        healthy=all_healthy,
        component="readiness",
        timestamp=time.time(),
        checks=dep_results,
        http_status=live.http_status,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    parser = argparse.ArgumentParser(description="MedVision-AI health probe")
    parser.add_argument("probe", choices=["liveness", "readiness"],
                        help="Which probe to run")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Service host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Service port (default: {DEFAULT_PORT})")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS,
                        help=f"Per-request timeout in seconds (default: {TIMEOUT_SECONDS})")
    args = parser.parse_args()

    if args.probe == "liveness":
        result = check_liveness(host=args.host, port=args.port, timeout=args.timeout)
    else:
        result = check_readiness(host=args.host, port=args.port, timeout=args.timeout)

    print(result.to_json())
    return 0 if result.healthy else 1


if __name__ == "__main__":
    sys.exit(_main())
