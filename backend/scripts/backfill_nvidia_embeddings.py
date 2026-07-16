#!/usr/bin/env python3
"""Backfill NVIDIA text embeddings into an OpenSearch kNN vector field.

The script is intentionally dry-run-first. It reads candidate documents by
default, but it only creates mappings, calls NVIDIA, or writes vectors when
`--execute` is passed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from openai import OpenAI  # noqa: E402
from opensearchpy import OpenSearch, helpers  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.opensearch_guardrails import parse_opensearch_endpoint  # noqa: E402

logger = logging.getLogger("backfill_nvidia_embeddings")

DEFAULT_NVIDIA_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_NVIDIA_DIMENSIONS = 384
DEFAULT_NVIDIA_VECTOR_FIELD = "dense_vector_nvidia_384"

SOURCE_FIELDS = [
    "image_id",
    "title",
    "description",
    "tags",
    "photographer",
]


@dataclass
class BackfillStats:
    scanned: int = 0
    embedded: int = 0
    updated: int = 0
    skipped_empty_text: int = 0
    failed: int = 0


def _as_clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _as_tag_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_as_clean_text(item) for item in value if _as_clean_text(item))
    return _as_clean_text(value)


def build_embedding_text(source: Dict[str, Any]) -> str:
    """Build deterministic passage text for document embeddings."""

    parts: List[str] = []
    title = _as_clean_text(source.get("title"))
    description = _as_clean_text(source.get("description"))
    tags = _as_tag_text(source.get("tags"))
    photographer = _as_clean_text(source.get("photographer"))

    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if tags:
        parts.append(f"Tags: {tags}")
    if photographer:
        parts.append(f"Photographer: {photographer}")
    return "\n".join(parts)


def build_vector_mapping(
    field_name: str,
    dimensions: int,
    *,
    engine: str = "lucene",
    space_type: str = "cosinesimil",
    ef_construction: int = 128,
    m: int = 24,
) -> Dict[str, Any]:
    return {
        "properties": {
            field_name: {
                "type": "knn_vector",
                "dimension": int(dimensions),
                "method": {
                    "name": "hnsw",
                    "space_type": space_type,
                    "engine": engine,
                    "parameters": {
                        "ef_construction": int(ef_construction),
                        "m": int(m),
                    },
                },
            }
        }
    }


def embedding_request_kwargs(
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int,
    input_type: str,
    truncate: Optional[str],
    send_dimensions: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": "float",
        "extra_body": {
            "input_type": input_type,
        },
    }
    if truncate:
        kwargs["extra_body"]["truncate"] = truncate
    if send_dimensions:
        kwargs["dimensions"] = int(dimensions)
    return kwargs


def chunked(items: Sequence[Tuple[str, str]], size: int) -> Iterator[List[Tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def create_opensearch_client() -> OpenSearch:
    endpoint = settings.opensearch_endpoint
    ep = parse_opensearch_endpoint(endpoint)
    parsed = urlparse(endpoint)
    client_kwargs: Dict[str, Any] = {
        "hosts": [{"host": ep.host, "port": ep.port}],
        "http_compress": True,
        "use_ssl": ep.scheme == "https",
        "verify_certs": False,
        "timeout": 60,
    }
    if parsed.username and parsed.password:
        client_kwargs["http_auth"] = (unquote(parsed.username), unquote(parsed.password))
    elif settings.opensearch_username and settings.opensearch_password:
        client_kwargs["http_auth"] = (
            settings.opensearch_username,
            settings.opensearch_password,
        )
    return OpenSearch(**client_kwargs)


def create_embedding_client(api_key: str, base_url: str, timeout_seconds: float) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
    )


def ensure_vector_mapping(
    client: OpenSearch,
    *,
    index: str,
    field_name: str,
    dimensions: int,
    engine: str,
    space_type: str,
    ef_construction: int,
    m: int,
    execute: bool,
) -> None:
    mapping = client.indices.get_mapping(index=index)
    index_mapping = mapping.get(index, {}).get("mappings", {})
    properties = index_mapping.get("properties", {})
    existing = properties.get(field_name)
    if existing:
        existing_dimension = int(existing.get("dimension", 0))
        if existing_dimension and existing_dimension != int(dimensions):
            raise RuntimeError(
                f"Existing field {field_name!r} has dimension {existing_dimension}, "
                f"not requested dimension {dimensions}"
            )
        logger.info("Vector field %s already exists on %s", field_name, index)
        return

    body = build_vector_mapping(
        field_name,
        dimensions,
        engine=engine,
        space_type=space_type,
        ef_construction=ef_construction,
        m=m,
    )
    if not execute:
        logger.info("Dry run: would create mapping on %s: %s", index, body)
        return

    logger.info("Creating vector mapping field=%s dimensions=%s on %s", field_name, dimensions, index)
    client.indices.put_mapping(index=index, body=body)


def iter_candidate_hits(
    client: OpenSearch,
    *,
    index: str,
    vector_field: str,
    scroll_size: int,
    scroll_ttl: str,
    limit: int,
    overwrite: bool,
) -> Iterator[Dict[str, Any]]:
    if overwrite:
        query: Dict[str, Any] = {"match_all": {}}
    else:
        query = {"bool": {"must_not": [{"exists": {"field": vector_field}}]}}

    body = {
        "size": int(scroll_size),
        "_source": SOURCE_FIELDS,
        "query": query,
    }
    response = client.search(index=index, body=body, scroll=scroll_ttl)
    scroll_id = response.get("_scroll_id")
    yielded = 0

    try:
        while True:
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                yield hit
                yielded += 1
                if limit and yielded >= limit:
                    return

            if not scroll_id:
                break
            response = client.scroll(scroll_id=scroll_id, scroll=scroll_ttl)
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to clear OpenSearch scroll: %s", exc)


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int,
    input_type: str,
    truncate: Optional[str],
    send_dimensions: bool,
) -> List[List[float]]:
    response = client.embeddings.create(
        **embedding_request_kwargs(
            texts,
            model=model,
            dimensions=dimensions,
            input_type=input_type,
            truncate=truncate,
            send_dimensions=send_dimensions,
        )
    )
    data = list(response.data or [])
    if len(data) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(data)}")

    vectors: List[List[float]] = []
    for item in data:
        vector = [float(value) for value in item.embedding]
        if len(vector) != int(dimensions):
            raise RuntimeError(
                f"Expected {dimensions}-dimensional embedding, got {len(vector)}"
            )
        vectors.append(vector)
    return vectors


def bulk_update_vectors(
    client: OpenSearch,
    *,
    index: str,
    vector_field: str,
    doc_vectors: Sequence[Tuple[str, List[float]]],
) -> int:
    actions = [
        {
            "_op_type": "update",
            "_index": index,
            "_id": doc_id,
            "doc": {
                vector_field: vector,
            },
        }
        for doc_id, vector in doc_vectors
    ]
    updated, _ = helpers.bulk(client, actions, request_timeout=120)
    return int(updated)


def resolve_api_key(cli_value: Optional[str]) -> str:
    api_key = (
        cli_value
        or settings.opensearch_text_embedding_api_key
        or settings.nvidia_api_key
    )
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY or OPENSEARCH_TEXT_EMBEDDING_API_KEY is required")
    return api_key


def process_batch(
    *,
    os_client: OpenSearch,
    embedding_client: Optional[OpenAI],
    batch: Sequence[Tuple[str, str]],
    index: str,
    vector_field: str,
    model: str,
    dimensions: int,
    input_type: str,
    truncate: Optional[str],
    send_dimensions: bool,
    execute: bool,
) -> Tuple[int, int]:
    if not batch:
        return 0, 0

    if not execute:
        logger.info("Dry run: would embed and update %s documents", len(batch))
        return 0, 0

    if embedding_client is None:
        raise RuntimeError("Embedding client is required when execute=True")

    doc_ids = [doc_id for doc_id, _ in batch]
    texts = [text for _, text in batch]
    vectors = embed_texts(
        embedding_client,
        texts,
        model=model,
        dimensions=dimensions,
        input_type=input_type,
        truncate=truncate,
        send_dimensions=send_dimensions,
    )
    updated = bulk_update_vectors(
        os_client,
        index=index,
        vector_field=vector_field,
        doc_vectors=list(zip(doc_ids, vectors)),
    )
    return len(vectors), updated


def run_backfill(args: argparse.Namespace) -> BackfillStats:
    os_client = create_opensearch_client()
    embedding_client = None
    if args.execute:
        embedding_client = create_embedding_client(
            api_key=resolve_api_key(args.api_key),
            base_url=args.base_url,
            timeout_seconds=args.embedding_timeout_seconds,
        )

    if args.create_mapping:
        ensure_vector_mapping(
            os_client,
            index=args.index,
            field_name=args.vector_field,
            dimensions=args.dimensions,
            engine=args.mapping_engine,
            space_type=args.mapping_space_type,
            ef_construction=args.mapping_ef_construction,
            m=args.mapping_m,
            execute=args.execute,
        )

    stats = BackfillStats()
    pending: List[Tuple[str, str]] = []
    effective_limit = args.limit
    if not args.execute and not effective_limit:
        effective_limit = args.dry_run_limit

    for hit in iter_candidate_hits(
        os_client,
        index=args.index,
        vector_field=args.vector_field,
        scroll_size=args.scroll_size,
        scroll_ttl=args.scroll_ttl,
        limit=effective_limit,
        overwrite=args.overwrite,
    ):
        stats.scanned += 1
        doc_id = str(hit.get("_id") or "")
        source = hit.get("_source") or {}
        text = build_embedding_text(source)
        if not doc_id or not text:
            stats.skipped_empty_text += 1
            continue

        if stats.scanned <= args.sample_text_count:
            logger.info("Sample doc_id=%s text=%r", doc_id, text[:500])

        pending.append((doc_id, text))
        if len(pending) >= args.batch_size:
            try:
                embedded, updated = process_batch(
                    os_client=os_client,
                    embedding_client=embedding_client,
                    batch=pending,
                    index=args.index,
                    vector_field=args.vector_field,
                    model=args.model,
                    dimensions=args.dimensions,
                    input_type=args.input_type,
                    truncate=args.truncate,
                    send_dimensions=args.send_dimensions,
                    execute=args.execute,
                )
                stats.embedded += embedded
                stats.updated += updated
            except Exception:
                stats.failed += len(pending)
                raise
            finally:
                pending = []

    if pending:
        try:
            embedded, updated = process_batch(
                os_client=os_client,
                embedding_client=embedding_client,
                batch=pending,
                index=args.index,
                vector_field=args.vector_field,
                model=args.model,
                dimensions=args.dimensions,
                input_type=args.input_type,
                truncate=args.truncate,
                send_dimensions=args.send_dimensions,
                execute=args.execute,
            )
            stats.embedded += embedded
            stats.updated += updated
        except Exception:
            stats.failed += len(pending)
            raise

    return stats


def default_model() -> str:
    if settings.normalized_embedding_provider == "nvidia":
        return settings.opensearch_text_embedding_model
    return DEFAULT_NVIDIA_MODEL


def default_dimensions() -> int:
    if settings.normalized_embedding_provider == "nvidia":
        return int(settings.opensearch_text_embedding_dimensions)
    return DEFAULT_NVIDIA_DIMENSIONS


def default_vector_field() -> str:
    if settings.normalized_embedding_provider == "nvidia":
        return settings.opensearch_vector_field
    return DEFAULT_NVIDIA_VECTOR_FIELD


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill NVIDIA passage embeddings into an OpenSearch vector field.",
    )
    parser.add_argument("--index", default=settings.opensearch_photo_index)
    parser.add_argument("--vector-field", default=default_vector_field())
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--dimensions", type=int, default=default_dimensions())
    parser.add_argument("--base-url", default=settings.nvidia_base_url)
    parser.add_argument("--api-key", default=None, help="Defaults to NVIDIA_API_KEY.")
    parser.add_argument("--input-type", default=settings.opensearch_text_embedding_passage_input_type or "passage")
    parser.add_argument("--truncate", default=settings.opensearch_text_embedding_truncate)
    parser.add_argument(
        "--no-send-dimensions",
        action="store_false",
        dest="send_dimensions",
        default=settings.opensearch_text_embedding_send_dimensions,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scroll-size", type=int, default=500)
    parser.add_argument("--scroll-ttl", default="5m")
    parser.add_argument("--limit", type=int, default=0, help="Maximum docs to scan; 0 means no limit.")
    parser.add_argument("--dry-run-limit", type=int, default=5)
    parser.add_argument("--sample-text-count", type=int, default=3)
    parser.add_argument("--embedding-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true", help="Re-embed docs even when the vector field exists.")
    parser.add_argument("--create-mapping", action="store_true")
    parser.add_argument("--mapping-engine", default="lucene")
    parser.add_argument("--mapping-space-type", default="cosinesimil")
    parser.add_argument("--mapping-ef-construction", type=int, default=128)
    parser.add_argument("--mapping-m", type=int, default=24)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call NVIDIA and write OpenSearch updates. Without this, the script is a dry run.",
    )
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    logger.info(
        "Backfill starting index=%s vector_field=%s model=%s dimensions=%s execute=%s",
        args.index,
        args.vector_field,
        args.model,
        args.dimensions,
        args.execute,
    )
    stats = run_backfill(args)
    logger.info("Backfill complete: %s", stats)
    if not args.execute:
        logger.info("Dry run only. Re-run with --execute to call NVIDIA and write vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
