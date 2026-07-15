"""
OpenSearch photo search service.
Searches the configured image index for stock photos matching user queries.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings
from app.services.opensearch_guardrails import create_opensearch_client, is_readonly_endpoint

logger = logging.getLogger(__name__)

LEXICAL_MINIMUM_SHOULD_MATCH = "75%"
BOOLEAN_CONNECTOR_TERMS = {"and", "or", "not"}
LEXICAL_TERM_RE = re.compile(r"\b[\w'-]+\b")


class PhotoSearchService:
    """Search stock photos in OpenSearch."""
    
    def __init__(self):
        """Initialize OpenSearch client."""
        readonly = is_readonly_endpoint(
            endpoint=settings.opensearch_endpoint,
            forced_readonly=settings.opensearch_readonly,
            readonly_hosts=settings.opensearch_readonly_hosts,
        )

        self.client = create_opensearch_client(
            endpoint=settings.opensearch_endpoint,
            readonly=readonly,
            timeout_seconds=30.0,
            username=settings.opensearch_username,
            password=settings.opensearch_password,
        )
        self.photo_index = settings.opensearch_photo_index
        self._embedding_client: Optional[OpenAI] = None
    
    def search_photos(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 20,
        min_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Search for stock photos matching the query.
        
        Args:
            query: Search query text
            filters: Optional filters (e.g., orientation, color, date_range)
            size: Maximum number of results to return
            min_score: Minimum relevance score threshold
            
        Returns:
            dict with keys:
                - results: List of photo documents
                - total: Total number of matching photos
                - took_ms: Search time in milliseconds
        """
        try:
            return self.execute_direct_hybrid_search(
                semantic_query=query,
                lexical_query=query,
                refinement_filters=self._legacy_filters_to_clauses(filters),
                size=size,
            )
            
        except Exception as e:
            logger.error(f"Photo search failed: {str(e)}", exc_info=True)
            return {
                'results': [],
                'total': 0,
                'took_ms': 0,
                'error': str(e)
            }

    def execute_direct_hybrid_search(
        self,
        semantic_query: str,
        lexical_query: Optional[str] = None,
        category_gids: Optional[List[int]] = None,
        exclusion_terms: Optional[List[str]] = None,
        refinement_filters: Optional[List[Dict[str, Any]]] = None,
        show_generated: bool = False,
        is_not_generated: bool = False,
        size: int = 50,
    ) -> Dict[str, Any]:
        """
        Execute an app-generated hybrid lexical + kNN query against icc_images_ext.
        """
        try:
            try:
                vector = self._embed_query_text(semantic_query)
            except Exception as embedding_error:
                logger.warning(
                    "Direct hybrid embedding generation failed; falling back to lexical search: %s",
                    embedding_error,
                )
                return self._execute_lexical_fallback_search(
                    semantic_query=semantic_query,
                    lexical_query=lexical_query or semantic_query,
                    category_gids=category_gids or [],
                    exclusion_terms=exclusion_terms or [],
                    refinement_filters=refinement_filters or [],
                    show_generated=show_generated,
                    is_not_generated=is_not_generated,
                    size=size,
                    error=str(embedding_error),
                )

            query_body = self.build_direct_hybrid_query(
                semantic_query=semantic_query,
                lexical_query=lexical_query or semantic_query,
                vector=vector,
                category_gids=category_gids or [],
                exclusion_terms=exclusion_terms or [],
                refinement_filters=refinement_filters or [],
                show_generated=show_generated,
                is_not_generated=is_not_generated,
                size=size,
            )

            params = {}
            if settings.opensearch_hybrid_search_pipeline:
                params["search_pipeline"] = settings.opensearch_hybrid_search_pipeline

            logger.info(
                "Executing direct hybrid OpenSearch query on index=%s semantic=%r lexical=%r",
                self.photo_index,
                semantic_query,
                lexical_query or semantic_query,
            )
            response = self.client.search(
                index=self.photo_index,
                body=query_body,
                params=params,
            )
            result = self._format_search_response(response)
            result["opensearch_query"] = query_body
            result["opensearch_pipeline"] = settings.opensearch_hybrid_search_pipeline
            result["opensearch_index"] = self.photo_index
            return result
        except Exception as e:
            logger.error("Direct hybrid query failed: %s", e, exc_info=True)
            return {
                "results": [],
                "total": 0,
                "took_ms": 0,
                "error": str(e),
            }

    def _execute_lexical_fallback_search(
        self,
        semantic_query: str,
        lexical_query: str,
        category_gids: Optional[List[int]] = None,
        exclusion_terms: Optional[List[str]] = None,
        refinement_filters: Optional[List[Dict[str, Any]]] = None,
        show_generated: bool = False,
        is_not_generated: bool = False,
        size: int = 50,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_body = self.build_direct_lexical_query(
            semantic_query=semantic_query,
            lexical_query=lexical_query,
            category_gids=category_gids or [],
            exclusion_terms=exclusion_terms or [],
            refinement_filters=refinement_filters or [],
            show_generated=show_generated,
            is_not_generated=is_not_generated,
            size=size,
        )
        response = self.client.search(
            index=self.photo_index,
            body=query_body,
            params={},
        )
        result = self._format_search_response(response)
        result["opensearch_query"] = query_body
        result["opensearch_pipeline"] = None
        result["opensearch_index"] = self.photo_index
        result["fallback"] = "lexical_only"
        if error:
            result["error"] = f"Embedding unavailable; used lexical fallback: {error}"
        return result

    def build_direct_hybrid_query(
        self,
        semantic_query: str,
        lexical_query: str,
        vector: List[float],
        category_gids: Optional[List[int]] = None,
        exclusion_terms: Optional[List[str]] = None,
        refinement_filters: Optional[List[Dict[str, Any]]] = None,
        show_generated: bool = False,
        is_not_generated: bool = False,
        size: int = 50,
    ) -> Dict[str, Any]:
        """
        Build the OpenSearch hybrid query for icc_images_ext.

        `icc_images_ext` exposes a 256-dim `dense_vector` kNN field plus text
        fields: title, description, and tags.
        """
        lexical_bool = self._build_lexical_bool(
            semantic_query=semantic_query,
            lexical_query=lexical_query,
            category_gids=category_gids or [],
            exclusion_terms=exclusion_terms or [],
            refinement_filters=refinement_filters or [],
            show_generated=show_generated,
            is_not_generated=is_not_generated,
        )

        return {
            "size": size,
            "_source": self._source_fields(),
            "query": {
                "hybrid": {
                    "queries": [
                        self._build_knn_clause(vector=vector, size=size),
                        {"bool": lexical_bool},
                    ]
                }
            },
        }

    def build_direct_lexical_query(
        self,
        semantic_query: str,
        lexical_query: str,
        category_gids: Optional[List[int]] = None,
        exclusion_terms: Optional[List[str]] = None,
        refinement_filters: Optional[List[Dict[str, Any]]] = None,
        show_generated: bool = False,
        is_not_generated: bool = False,
        size: int = 50,
    ) -> Dict[str, Any]:
        return {
            "size": size,
            "_source": self._source_fields(),
            "query": {
                "bool": self._build_lexical_bool(
                    semantic_query=semantic_query,
                    lexical_query=lexical_query,
                    category_gids=category_gids or [],
                    exclusion_terms=exclusion_terms or [],
                    refinement_filters=refinement_filters or [],
                    show_generated=show_generated,
                    is_not_generated=is_not_generated,
                )
            },
        }

    def _source_fields(self) -> List[str]:
        return [
            "image_id",
            "title",
            "description",
            "tags",
            "thumbnail_url",
            "medium_url",
            "pexels_url",
            "photographer",
            "width",
            "height",
        ]

    def _meaningful_lexical_terms(self, query: str) -> List[str]:
        return [
            term
            for term in LEXICAL_TERM_RE.findall(query or "")
            if term.lower() not in BOOLEAN_CONNECTOR_TERMS
        ]

    def _apply_lexical_match_policy(
        self,
        multi_match: Dict[str, Any],
        query: str,
    ) -> None:
        term_count = len(self._meaningful_lexical_terms(query))
        multi_match.pop("operator", None)
        multi_match.pop("minimum_should_match", None)

        if term_count >= 4:
            multi_match["minimum_should_match"] = LEXICAL_MINIMUM_SHOULD_MATCH
        else:
            multi_match["operator"] = "and"

    def _build_lexical_bool(
        self,
        semantic_query: str,
        lexical_query: str,
        category_gids: List[int],
        exclusion_terms: List[str],
        refinement_filters: List[Dict[str, Any]],
        show_generated: bool,
        is_not_generated: bool,
    ) -> Dict[str, Any]:
        query_text = lexical_query or semantic_query
        multi_match: Dict[str, Any] = {
            "query": query_text,
            "fields": [
                "title^4",
                "description^3",
                "tags^2",
                "photographer",
            ],
            "type": "best_fields",
            "fuzziness": "AUTO",
        }
        self._apply_lexical_match_policy(multi_match, query_text)

        lexical_bool: Dict[str, Any] = {
            "must": [
                {
                    "multi_match": multi_match
                }
            ]
        }

        must_not = self._build_text_exclusions(exclusion_terms or [])
        if must_not:
            lexical_bool["must_not"] = must_not

        filters = self._supported_filter_clauses(
            category_gids=category_gids or [],
            refinement_filters=refinement_filters or [],
            show_generated=show_generated,
            is_not_generated=is_not_generated,
        )
        if filters:
            lexical_bool["filter"] = filters
        return lexical_bool

    def _build_knn_clause(self, vector: List[float], size: int) -> Dict[str, Any]:
        vector_query: Dict[str, Any] = {"vector": vector}
        if settings.opensearch_knn_min_score > 0:
            vector_query["min_score"] = float(settings.opensearch_knn_min_score)
        else:
            vector_query["k"] = max(int(settings.opensearch_knn_k), size)
        return {"knn": {settings.opensearch_vector_field: vector_query}}
    
    def execute_raw_query(
        self,
        opensearch_query: Dict[str, Any],
        index: str = None,
        search_pipeline: str = "hybrid-rrf-60",
    ) -> Dict[str, Any]:
        """
        Execute a raw OpenSearch query DSL (e.g. from Search Service debug.request).
        
        Args:
            opensearch_query: Raw OpenSearch query body
            index: Index to search (defaults to self.photo_index)
            
        Returns:
            dict with results, total, took_ms
        """
        target_index = index or self.photo_index
        
        try:
            logger.info(f"Executing raw OpenSearch query on index: {target_index}")
            
            response = self.client.search(
                index=target_index,
                body=opensearch_query,
                params={"search_pipeline": search_pipeline},
            )

            result = self._format_search_response(response)
            logger.info(
                "Raw query: Found %s photos in %sms, returning %s results",
                result["total"],
                result["took_ms"],
                len(result["results"]),
            )
            return result
            
        except Exception as e:
            logger.error(f"Raw query execution failed: {str(e)}", exc_info=True)
            return {
                'results': [],
                'total': 0,
                'took_ms': 0,
                'error': str(e)
            }

    def _legacy_filters_to_clauses(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not filters:
            return []
        clauses: List[Dict[str, Any]] = []
        if "orientation" in filters:
            clauses.append({"term": {"orientation": filters["orientation"]}})
        if "min_license_count" in filters:
            clauses.append(
                {
                    "range": {
                        "total_paid_license_count_all_time": {
                            "gte": filters["min_license_count"]
                        }
                    }
                }
            )
        if "date_from" in filters or "date_to" in filters:
            date_range = {}
            if "date_from" in filters:
                date_range["gte"] = filters["date_from"]
            if "date_to" in filters:
                date_range["lte"] = filters["date_to"]
            clauses.append({"range": {"date_added": date_range}})
        return clauses

    def _build_search_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        size: int,
        min_score: float
    ) -> Dict[str, Any]:
        """Build OpenSearch query DSL."""
        
        # Multi-match query across description and keywords
        must_clauses = [
            {
                'multi_match': {
                    'query': query,
                    'fields': [
                        'description_en^3',  # Boost description
                        'keywords^2',        # Boost keywords
                        'categories'
                    ],
                    'type': 'best_fields',
                    'operator': 'or',
                    'fuzziness': 'AUTO'
                }
            }
        ]
        
        # Add filters if provided
        filter_clauses = []
        if filters:
            if 'orientation' in filters:
                filter_clauses.append({
                    'term': {'orientation': filters['orientation']}
                })
            
            if 'min_license_count' in filters:
                filter_clauses.append({
                    'range': {
                        'total_paid_license_count_all_time': {
                            'gte': filters['min_license_count']
                        }
                    }
                })
            
            if 'date_from' in filters or 'date_to' in filters:
                date_range = {}
                if 'date_from' in filters:
                    date_range['gte'] = filters['date_from']
                if 'date_to' in filters:
                    date_range['lte'] = filters['date_to']
                
                filter_clauses.append({
                    'range': {'date_added': date_range}
                })
            
            if 'categories' in filters and filters['categories']:
                filter_clauses.append({
                    'terms': {'categories': filters['categories']}
                })
        
        # Construct query
        bool_query = {
            'must': must_clauses
        }
        
        if filter_clauses:
            bool_query['filter'] = filter_clauses
        
        search_body = {
            'query': {
                'bool': bool_query
            },
            'size': size,
            'min_score': min_score,
            '_source': [
                'hadron_id',
                'ext_id',
                'description_en',
                'date_added',
                'total_paid_license_count_all_time',
                'categories',
                'keywords',
                'orientation'
            ],
            'sort': [
                {'_score': {'order': 'desc'}},
                {'total_paid_license_count_all_time': {'order': 'desc'}}
            ]
        }
        
        return search_body

    def _embed_query_text(self, query: str) -> List[float]:
        clean_query = " ".join(str(query or "").split())
        if not clean_query:
            raise RuntimeError("Cannot generate embedding for an empty query")

        response = self._get_embedding_client().embeddings.create(
            model=settings.opensearch_text_embedding_model,
            input=clean_query,
            dimensions=settings.opensearch_text_embedding_dimensions,
            encoding_format="float",
        )
        if not response.data:
            raise RuntimeError("No embedding generated for query")

        vector = [float(v) for v in response.data[0].embedding]
        expected_dimensions = int(settings.opensearch_text_embedding_dimensions)
        if len(vector) != expected_dimensions:
            raise RuntimeError(
                f"Expected {expected_dimensions}-dimensional embedding, got {len(vector)}"
            )
        return vector

    def _get_embedding_client(self) -> OpenAI:
        if self._embedding_client is not None:
            return self._embedding_client

        client_kwargs: Dict[str, Any] = {
            "api_key": settings.require_openai_api_key(),
            "timeout": settings.opensearch_text_embedding_timeout_seconds,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._embedding_client = OpenAI(**client_kwargs)
        return self._embedding_client

    def _supported_filter_clauses(
        self,
        category_gids: List[int],
        refinement_filters: List[Dict[str, Any]],
        show_generated: bool,
        is_not_generated: bool,
    ) -> List[Dict[str, Any]]:
        # icc_images_ext currently has no category/orientation/date/generated
        # fields. Keep the method explicit so unsupported filters are ignored
        # rather than producing zero-result queries against unmapped fields.
        return []

    def _build_text_exclusions(self, exclusion_terms: List[str]) -> List[Dict[str, Any]]:
        clauses: List[Dict[str, Any]] = []
        for term in exclusion_terms:
            text = str(term).strip()
            if not text:
                continue
            clauses.append(
                {
                    "multi_match": {
                        "query": text,
                        "fields": ["title", "description", "tags"],
                        "type": "best_fields",
                    }
                }
            )
        return clauses

    def _format_search_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        hits = response.get("hits", {}).get("hits", [])
        total_raw = response.get("hits", {}).get("total", {})
        total = total_raw.get("value", 0) if isinstance(total_raw, dict) else int(total_raw or 0)
        took_ms = response.get("took", 0)

        results = [self._format_hit(hit) for hit in hits]
        logger.info(
            "OpenSearch: Found %s photos in %sms, returning %s results",
            total,
            took_ms,
            len(results),
        )
        return {
            "results": results,
            "total": total,
            "took_ms": took_ms,
        }

    def _format_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        source = hit.get("_source", {})
        fields = hit.get("fields", {})

        image_id = (
            source.get("image_id")
            or source.get("ext_id")
            or (fields.get("image_id", [None])[0] if fields.get("image_id") else None)
            or (fields.get("ext_id", [None])[0] if fields.get("ext_id") else None)
        )
        ext_id = self._safe_int(image_id)
        tags = source.get("tags") or source.get("keywords") or source.get("keywords_en") or []
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]

        media_type = source.get("media_type", "image")
        source_page_url = source.get("pexels_url") or source.get("image_url")
        display_image_url = (
            source.get("medium_url")
            or source.get("thumbnail_url")
            or source.get("image_url")
            or source_page_url
        )
        image_url = (
            source_page_url
            or source.get("medium_url")
            or source.get("thumbnail_url")
            or self._build_image_url(ext_id, media_type)
        )
        thumbnail_url = (
            display_image_url
            or self._build_thumbnail_url(ext_id, media_type)
        )

        return {
            "hadron_id": str(image_id) if image_id is not None else None,
            "ext_id": ext_id,
            "title": source.get("title"),
            "description": source.get("description") or source.get("description_en") or source.get("title") or "No description available",
            "image_url": image_url,
            "thumbnail_url": thumbnail_url,
            "video_url": source.get("video_url") or self._build_video_url(ext_id, media_type),
            "media_type": media_type,
            "date_added": source.get("date_added"),
            "license_count": source.get("total_paid_license_count_all_time", 0),
            "categories": source.get("categories") or source.get("global_category_ids", []),
            "keywords": tags,
            "tags": tags,
            "orientation": source.get("orientation"),
            "score": hit.get("_score", 0.0),
            "is_generated": bool(source.get("is_generated", False)),
            "photographer": source.get("photographer"),
            "pexels_url": source.get("pexels_url"),
            "width": source.get("width"),
            "height": source.get("height"),
        }

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(str(value))
        except Exception:
            return None
    
    def _build_image_url(self, ext_id: str, media_type: str = 'image') -> str:
        """Build full-size image/preview URL from ext_id."""
        if not ext_id:
            return ''
        if media_type == 'video':
            return f"http://localhost:9200/assets/videos/{ext_id}/thumb/1.jpg"
        return f"http://localhost:9200/assets/image-250nw-{ext_id}.jpg"

    def _build_thumbnail_url(self, ext_id: str, media_type: str = 'image') -> str:
        """Build thumbnail URL from ext_id."""
        if not ext_id:
            return ''
        if media_type == 'video':
            return f"http://localhost:9200/assets/videos/{ext_id}/thumb/1.jpg"
        return f"http://localhost:9200/assets/image-150nw-{ext_id}.jpg"

    def _build_video_url(self, ext_id: str, media_type: str = 'image') -> str:
        """Build MP4 preview URL for video assets."""
        if not ext_id or media_type != 'video':
            return ''
        return f"http://localhost:9200/assets/videos/{ext_id}/preview/_.mp4"


# Singleton instance
photo_search_service = PhotoSearchService()
