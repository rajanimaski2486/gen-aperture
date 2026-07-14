"""OpenSearch client creation with strict read-only guardrails.

Goal: protect production OpenSearch clusters from any write operations.

We enforce an allow-list of HTTP methods and endpoints when read-only mode
is active:
- GET / HEAD: allowed
- POST: allowed only for search endpoints (e.g. */_search)

Anything else (PUT/PATCH/DELETE or POST to write-ish endpoints like _bulk)
raises PermissionError before the request is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

from opensearchpy import OpenSearch


@dataclass(frozen=True)
class OpenSearchEndpoint:
    scheme: str
    host: str
    port: int


def parse_opensearch_endpoint(endpoint: str) -> OpenSearchEndpoint:
    parsed = urlparse(endpoint)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or endpoint.replace("http://", "").replace("https://", "").split(":")[0]
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return OpenSearchEndpoint(scheme=scheme, host=host, port=port)


def is_readonly_endpoint(endpoint: str, forced_readonly: Optional[bool], readonly_hosts: list[str]) -> bool:
    parsed = urlparse(endpoint)
    host = parsed.hostname or ""
    # Guardrail rule: if the endpoint host is in the read-only host list,
    # we ALWAYS enforce read-only (no environment override to disable).
    if host in set(readonly_hosts):
        return True

    if forced_readonly is not None:
        return bool(forced_readonly)

    return False


def _normalize_path(url: str) -> str:
    # opensearchpy passes URLs like "/index/_search".
    # Be resilient to accidental full URLs.
    parsed = urlparse(url)
    path = parsed.path or url
    if not path.startswith("/"):
        path = "/" + path
    return path


def _path_segments(url: str) -> list[str]:
    path = _normalize_path(url)
    return [unquote(segment) for segment in path.strip("/").split("/") if segment]


def _readonly_request_allowed(method: str, url: str) -> bool:
    method_upper = (method or "").upper()
    path = _normalize_path(url)

    if method_upper in {"GET", "HEAD"}:
        return True

    if method_upper == "POST":
        # Allow only search-style POST endpoints.
        # Intentionally *not* allowing _msearch/_count for maximum safety.
        if path.endswith("/_search"):
            return True
        # Scroll is read-only but uses POST.
        if path.startswith("/_search/scroll") or path.endswith("/_search/scroll"):
            return True

    return False


def install_readonly_guardrails(client: OpenSearch) -> None:
    """Monkey-patch the OpenSearch transport to block any non-read requests."""

    transport = client.transport
    original_perform_request: Callable = transport.perform_request

    def guarded_perform_request(method, url, params=None, body=None, headers=None):  # type: ignore[no-untyped-def]
        if not _readonly_request_allowed(method, url):
            path = _normalize_path(url)
            raise PermissionError(
                "OpenSearch prod guardrails: blocked non-read request "
                f"method={str(method).upper()} path={path}"
            )
        return original_perform_request(method, url, params=params, body=body, headers=headers)

    transport.perform_request = guarded_perform_request  # type: ignore[assignment]


def _conversation_request_allowed(method: str, url: str, allowed_index: str) -> bool:
    method_upper = (method or "").upper()
    segments = _path_segments(url)

    if method_upper in {"GET", "HEAD"}:
        return True

    # Search and count APIs are read operations that may use POST.
    if method_upper == "POST" and segments:
        if segments[-1] in {"_search", "_count"}:
            return True
        if segments[0] == "_search" and len(segments) >= 2 and segments[1] == "scroll":
            return True

    if not segments or segments[0] != allowed_index:
        return False

    # Allow creating the one conversation index. The caller checks existence
    # before issuing this request; the guardrail keeps the index allow-list
    # narrow even if another caller reaches this client.
    if method_upper == "PUT" and len(segments) == 1:
        return True

    # Allow document-level conversation mutations only within the configured
    # index. Whole-index deletes, mappings/settings changes, aliases, bulk
    # writes, and cluster/plugin writes remain blocked.
    if len(segments) >= 2:
        operation = segments[1]
        if operation in {"_doc", "_create"} and method_upper in {"PUT", "POST", "DELETE"}:
            return True
        if operation == "_update" and method_upper == "POST":
            return True

    return False


def install_conversation_write_guardrails(client: OpenSearch, allowed_index: str) -> None:
    """Allow writes only to the configured conversation index."""

    transport = client.transport
    original_perform_request: Callable = transport.perform_request

    def guarded_perform_request(method, url, params=None, body=None, headers=None):  # type: ignore[no-untyped-def]
        if not _conversation_request_allowed(method, url, allowed_index):
            path = _normalize_path(url)
            raise PermissionError(
                "OpenSearch guardrails: blocked write outside allowed "
                f"conversation index={allowed_index} method={str(method).upper()} path={path}"
            )
        return original_perform_request(method, url, params=params, body=body, headers=headers)

    transport.perform_request = guarded_perform_request  # type: ignore[assignment]


def create_opensearch_client(
    endpoint: str,
    readonly: bool,
    timeout_seconds: float = 30.0,
    username: Optional[str] = None,
    password: Optional[str] = None,
    allowed_write_index: Optional[str] = None,
) -> OpenSearch:
    ep = parse_opensearch_endpoint(endpoint)
    parsed = urlparse(endpoint)

    client_kwargs = {
        "hosts": [{"host": ep.host, "port": ep.port}],
        "http_compress": True,
        "use_ssl": (ep.scheme == "https"),
        "verify_certs": False,
        "timeout": timeout_seconds,
    }

    if parsed.username and parsed.password:
        client_kwargs["http_auth"] = (unquote(parsed.username), unquote(parsed.password))
    elif username and password:
        client_kwargs["http_auth"] = (username, password)

    client = OpenSearch(**client_kwargs)

    if readonly:
        install_readonly_guardrails(client)
    elif allowed_write_index:
        install_conversation_write_guardrails(client, allowed_write_index)

    return client
