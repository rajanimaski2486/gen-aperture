import unittest

from scripts.backfill_nvidia_embeddings import (
    build_embedding_text,
    build_vector_mapping,
    embedding_request_kwargs,
)


class NvidiaEmbeddingBackfillTests(unittest.TestCase):
    def test_build_embedding_text_uses_searchable_fields(self):
        text = build_embedding_text(
            {
                "title": " Mountain sunrise ",
                "description": "  Bright light over the valley  ",
                "tags": ["mountain", " sunrise ", ""],
                "photographer": "Ada",
            }
        )

        self.assertEqual(
            text,
            "\n".join(
                [
                    "Title: Mountain sunrise",
                    "Description: Bright light over the valley",
                    "Tags: mountain, sunrise",
                    "Photographer: Ada",
                ]
            ),
        )

    def test_build_vector_mapping_is_dimension_aware(self):
        mapping = build_vector_mapping("dense_vector_nvidia_384", 384)

        field = mapping["properties"]["dense_vector_nvidia_384"]
        self.assertEqual(field["type"], "knn_vector")
        self.assertEqual(field["dimension"], 384)
        self.assertEqual(field["method"]["space_type"], "cosinesimil")
        self.assertEqual(field["method"]["engine"], "lucene")

    def test_embedding_request_kwargs_uses_passage_input_type(self):
        kwargs = embedding_request_kwargs(
            ["Title: Mountain sunrise"],
            model="nvidia/llama-nemotron-embed-1b-v2",
            dimensions=384,
            input_type="passage",
            truncate="END",
            send_dimensions=True,
        )

        self.assertEqual(kwargs["model"], "nvidia/llama-nemotron-embed-1b-v2")
        self.assertEqual(kwargs["input"], ["Title: Mountain sunrise"])
        self.assertEqual(kwargs["dimensions"], 384)
        self.assertEqual(kwargs["extra_body"], {"input_type": "passage", "truncate": "END"})


if __name__ == "__main__":
    unittest.main()
