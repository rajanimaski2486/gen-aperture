import unittest

from app.config import settings
from app.services.photo_search import PhotoSearchService


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    def search(self, index, body, params=None):
        self.calls.append({"index": index, "body": body, "params": params or {}})
        return {
            "took": 7,
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 1.25,
                        "_source": {
                            "image_id": "12345",
                            "title": "Mountain sunrise",
                            "description": "A bright sunrise over mountains",
                            "tags": ["mountain", "sunrise"],
                            "thumbnail_url": "https://example.test/thumb.jpg",
                            "medium_url": "https://example.test/medium.jpg",
                            "pexels_url": "https://example.test/photo",
                            "photographer": "Ada",
                            "width": 1200,
                            "height": 800,
                        },
                    }
                ],
            },
        }


def make_service():
    service = object.__new__(PhotoSearchService)
    service.photo_index = settings.opensearch_photo_index
    service.client = FakeSearchClient()
    service._text_embedder = None
    service._text_embedder_pca_path = None
    service._embed_query_text = lambda query: [0.01] * 256
    return service


class DirectPhotoSearchTests(unittest.TestCase):
    def test_build_direct_hybrid_query_combines_knn_and_lexical(self):
        service = make_service()
        body = service.build_direct_hybrid_query(
            semantic_query="sunrise mountains",
            lexical_query="sunrise AND mountains",
            vector=[0.01] * 256,
            exclusion_terms=["snow"],
            size=25,
        )

        queries = body["query"]["hybrid"]["queries"]
        self.assertIn("knn", queries[0])
        self.assertIn(settings.opensearch_vector_field, queries[0]["knn"])
        self.assertEqual(queries[0]["knn"][settings.opensearch_vector_field]["k"], settings.opensearch_knn_k)

        lexical = queries[1]["bool"]
        multi_match = lexical["must"][0]["multi_match"]
        self.assertEqual(multi_match["query"], "sunrise AND mountains")
        self.assertIn("title^4", multi_match["fields"])
        self.assertIn("description^3", multi_match["fields"])
        self.assertIn("tags^2", multi_match["fields"])
        self.assertIn("must_not", lexical)

    def test_execute_direct_hybrid_search_maps_icc_image_fields(self):
        service = make_service()

        result = service.execute_direct_hybrid_search(
            semantic_query="mountain sunrise",
            lexical_query="mountain sunrise",
            size=10,
        )

        self.assertEqual(service.client.calls[0]["index"], settings.opensearch_photo_index)
        self.assertEqual(
            service.client.calls[0]["params"]["search_pipeline"],
            settings.opensearch_hybrid_search_pipeline,
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["took_ms"], 7)

        photo = result["results"][0]
        self.assertEqual(photo["hadron_id"], "12345")
        self.assertEqual(photo["ext_id"], 12345)
        self.assertEqual(photo["description"], "A bright sunrise over mountains")
        self.assertEqual(photo["image_url"], "https://example.test/medium.jpg")
        self.assertEqual(photo["thumbnail_url"], "https://example.test/thumb.jpg")
        self.assertEqual(photo["keywords"], ["mountain", "sunrise"])
        self.assertEqual(photo["photographer"], "Ada")
        self.assertIn("opensearch_query", result)


if __name__ == "__main__":
    unittest.main()
