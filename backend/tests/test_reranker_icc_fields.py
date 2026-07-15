import asyncio
import json
import unittest

from app.config import Settings
from app.services.reranker import (
    ReflectionReranker,
    RerankerConfig,
    _build_candidate_summary,
    _format_criteria,
    _infer_orientation,
    _python_dedup,
)


class RerankerIccFieldsTests(unittest.TestCase):
    def test_candidate_summary_uses_only_supported_icc_fields(self):
        summary = json.loads(
            _build_candidate_summary(
                [
                    {
                        "hadron_id": "34194567",
                        "title": "Industrial robotic arm",
                        "description": "Close-up of a Delta brand robotic arm in an automated setup.",
                        "keywords": ["robot", "automation"],
                        "width": 4000,
                        "height": 6000,
                        "score": 0.87654,
                        "license_count": 99,
                        "category_ids": [123],
                        "photographer": "Ada",
                        "pexels_url": "https://www.pexels.com/photo/example/",
                    }
                ]
            )
        )

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(
            set(row),
            {
                "index",
                "hadron_id",
                "title",
                "description",
                "tags",
                "width",
                "height",
                "orientation",
                "retrieval_score",
            },
        )
        self.assertEqual(row["tags"], ["robot", "automation"])
        self.assertEqual(row["orientation"], "vertical")
        self.assertEqual(row["retrieval_score"], 0.8765)

    def test_orientation_is_inferred_from_dimensions(self):
        self.assertEqual(_infer_orientation(4000, 2600), "horizontal")
        self.assertEqual(_infer_orientation(2600, 4000), "vertical")
        self.assertEqual(_infer_orientation(3000, 3020), "square")
        self.assertIsNone(_infer_orientation(None, 3020))

    def test_python_dedup_uses_title_description_tags_not_legacy_keywords_only(self):
        candidates = {
            "strong": {
                "title": "Close-up industrial robotic arm",
                "description": "Industrial robotic arm in automated manufacturing setup.",
                "tags": [],
                "score": 2.0,
            },
            "duplicate": {
                "title": "Close-up industrial robotic arm",
                "description": "Industrial robotic arm in automated manufacturing setup.",
                "tags": [],
                "score": 1.0,
            },
        }

        self.assertEqual(_python_dedup(candidates, threshold=0.5), ["duplicate"])

    def test_format_criteria_ignores_unsupported_legacy_filter_fields(self):
        criteria = _format_criteria(
            {
                "user_query": "horizontal robotic arm",
                "exclusion_terms": ["cartoon"],
                "refinement_filters": [
                    {"term": {"orientation": "horizontal"}},
                    {"range": {"date_added": {"gte": "2026-01-01"}}},
                    {"range": {"total_paid_license_count_all_time": {"gte": 10}}},
                    {"range": {"width": {"gte": 1000}}},
                ],
                "category_gids": [123],
            }
        )

        self.assertIn('Primary search subject: "horizontal robotic arm"', criteria)
        self.assertIn("title, description, or tags", criteria)
        self.assertIn("orientation=horizontal", criteria)
        self.assertIn("width gte:1000", criteria)
        self.assertNotIn("date_added", criteria)
        self.assertNotIn("total_paid_license_count_all_time", criteria)
        self.assertNotIn("Category GIDs", criteria)

    def test_reranker_defaults_to_fast_model_and_timeout(self):
        self.assertEqual(
            Settings.model_fields["rerank_model"].default,
            "meta/llama-3.2-3b-instruct",
        )
        self.assertEqual(
            Settings.model_fields["rerank_timeout_seconds"].default,
            120.0,
        )

    def test_reranker_timeout_returns_original_candidates(self):
        candidates = [
            {
                "hadron_id": "rose-1",
                "title": "Red rose",
                "description": "A dramatic red rose.",
                "tags": ["rose"],
                "score": 0.9,
            }
        ]
        reranker = ReflectionReranker(
            RerankerConfig(
                model="meta/llama-3.2-3b-instruct",
                timeout_seconds=0.01,
                api_key="test-key",
                base_url="https://example.invalid/v1",
            )
        )

        async def slow_scoring_pass(client, user_query, criteria_text, candidates):
            await asyncio.sleep(0.1)
            return []

        reranker._scoring_pass = slow_scoring_pass

        result = asyncio.run(reranker.rerank("best rose", candidates))

        self.assertFalse(result.triggered)
        self.assertEqual(result.final_results, candidates)
        self.assertIn("timed out", result.explanation)
        self.assertEqual(result.pass_summaries["error"], "timeout")


if __name__ == "__main__":
    unittest.main()
