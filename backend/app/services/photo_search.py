"""
OpenSearch photo search service.
Searches the web-index-v9 index for stock photos matching user queries.
"""
import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.opensearch_guardrails import create_opensearch_client, is_readonly_endpoint

logger = logging.getLogger(__name__)


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
        )
        self.photo_index = settings.opensearch_photo_index
    
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
            # Build OpenSearch query
            search_body = self._build_search_query(query, filters, size, min_score)
            
            logger.info(f"Searching photos with query: {query}, filters: {filters}")
            
            # Execute search
            response = self.client.search(
                index=self.photo_index,
                body=search_body,
                params={"search_pipeline": "hybrid-rrf-60"}
            )
            
            # Parse results
            hits = response.get('hits', {}).get('hits', [])
            total = response.get('hits', {}).get('total', {}).get('value', 0)
            took_ms = response.get('took', 0)
            
            # Format results
            results = []
            for hit in hits:
                source = hit.get('_source', {})
                media_type = source.get('media_type', 'image')
                results.append({
                    'hadron_id': source.get('hadron_id'),
                    'ext_id': source.get('ext_id'),
                    'description': source.get('description_en', 'No description available'),
                    'image_url': self._build_image_url(source.get('ext_id'), media_type),
                    'thumbnail_url': self._build_thumbnail_url(source.get('ext_id'), media_type),
                    'date_added': source.get('date_added'),
                    'license_count': source.get('total_paid_license_count_all_time', 0),
                    'categories': source.get('categories', []),
                    'keywords': source.get('keywords', []),
                    'orientation': source.get('orientation'),
                    'score': hit.get('_score', 0.0)
                })
            
            logger.info(f"Found {total} photos in {took_ms}ms, returning {len(results)} results")
            
            return {
                'results': results,
                'total': total,
                'took_ms': took_ms
            }
            
        except Exception as e:
            logger.error(f"Photo search failed: {str(e)}", exc_info=True)
            return {
                'results': [],
                'total': 0,
                'took_ms': 0,
                'error': str(e)
            }
    
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

            # Parse results
            hits = response.get('hits', {}).get('hits', [])
            total = response.get('hits', {}).get('total', {}).get('value', 0)
            took_ms = response.get('took', 0)
            
            # Format results
            results = []
            for hit in hits:
                source = hit.get('_source', {})
                # Also check docvalue_fields in the hit (fields key)
                fields = hit.get('fields', {})
                
                # ext_id can be in _source or in fields
                ext_id = source.get('ext_id') or (fields.get('ext_id', [None])[0] if fields.get('ext_id') else None)
                hadron_id = source.get('hadron_id') or (fields.get('hadron_id', [None])[0] if fields.get('hadron_id') else None)
                media_type = source.get('media_type', 'image')
                is_generated_raw = source.get('is_generated')
                if is_generated_raw is None and fields.get('is_generated'):
                    is_generated_raw = fields['is_generated'][0]
                is_generated = bool(is_generated_raw) if is_generated_raw is not None else False
                
                results.append({
                    'hadron_id': hadron_id,
                    'ext_id': ext_id,
                    'description': source.get('description_en', 'No description available'),
                    'image_url': self._build_image_url(ext_id, media_type),
                    'thumbnail_url': self._build_thumbnail_url(ext_id, media_type),
                    'video_url': self._build_video_url(ext_id, media_type),
                    'media_type': media_type,
                    'date_added': source.get('date_added'),
                    'license_count': source.get('total_paid_license_count_all_time', 0),
                    'categories': source.get('categories') or source.get('global_category_ids', []),
                    'keywords': source.get('keywords') or source.get('keywords_en', []),
                    'orientation': source.get('orientation'),
                    'score': hit.get('_score', 0.0),
                    'is_generated': is_generated,
                })
            
            logger.info(f"Raw query: Found {total} photos in {took_ms}ms, returning {len(results)} results")
            
            return {
                'results': results,
                'total': total,
                'took_ms': took_ms
            }
            
        except Exception as e:
            logger.error(f"Raw query execution failed: {str(e)}", exc_info=True)
            return {
                'results': [],
                'total': 0,
                'took_ms': 0,
                'error': str(e)
            }

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
