import unittest

from app.services.agent_squad import _detect_search_mode
from app.services.query_intent import detect_text_query_intent


class ExplodingLlm:
    def invoke(self, messages):
        raise AssertionError("LLM should not be called for default text intent")


class DummyCategoryFilter:
    _gid_to_value = {}

    def label_for_gid(self, gid):
        return str(gid)

    def match_categories(self, name):
        return []


class AgentSquadLatencyTests(unittest.TestCase):
    def test_best_match_query_is_relevance_not_popular(self):
        self.assertEqual(_detect_search_mode("best roses please"), "relevance")
        self.assertEqual(_detect_search_mode("best matching rose photos"), "relevance")

    def test_explicit_popularity_queries_are_popular(self):
        self.assertEqual(_detect_search_mode("most popular rose photos"), "popular")
        self.assertEqual(_detect_search_mode("trending summer beverage images"), "popular")
        self.assertEqual(_detect_search_mode("best-selling travel images"), "popular")

    def test_text_intent_uses_fast_local_path_by_default(self):
        result = detect_text_query_intent(
            "best roses please",
            ExplodingLlm(),
            DummyCategoryFilter(),
        )

        self.assertEqual(result.intent, "Direct search (fast path)")
        self.assertEqual(result.entity_terms, ["roses"])
        self.assertEqual(result.boolean_query, "roses")
        self.assertEqual(result.semantic_query, "roses")

    def test_text_intent_fast_path_extracts_basic_constraints(self):
        result = detect_text_query_intent(
            "horizontal rose photos without people",
            ExplodingLlm(),
            DummyCategoryFilter(),
        )

        self.assertEqual(result.media_type, "image")
        self.assertIn("people", result.exclusion_terms)
        self.assertTrue(result.refinement_filters)


if __name__ == "__main__":
    unittest.main()
