import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import settings
from app.services.agent_squad import AgentSquad, _detect_search_mode
from app.services.query_intent import (
    build_contextual_query_fallback,
    detect_text_query_intent,
)


class ExplodingLlm:
    def invoke(self, messages):
        raise AssertionError("LLM should not be called for default text intent")


class FailingResolverLlm:
    def invoke(self, messages):
        raise RuntimeError("resolver unavailable")


class WeakAnaphoraResolverLlm:
    def invoke(self, messages):
        return SimpleNamespace(content="blue ones")


class JsonIntentLlm:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="""{
            "intent": "Find red roses without people",
            "entity_terms": ["red", "roses"],
            "named_entities": {
                "locations": [],
                "brands_trademarks": [],
                "celebrities": [],
                "seasons": []
            },
            "media_type": "image",
            "is_generated": null,
            "filters": {
                "orientation": null,
                "recency_gte": null,
                "popularity_gte": null
            },
            "exclusion_terms": ["people"],
            "boolean_query": "red AND roses",
            "expanded_semantic_query": "red roses floral bouquet",
            "suggested_categories": [],
            "mood_style": []
        }""")


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

    def test_contextual_query_fallback_preserves_previous_user_search_for_followup(self):
        history = [
            {"role": "user", "content": "red roses"},
            {"role": "assistant", "content": "I found red rose images."},
        ]

        self.assertEqual(
            build_contextual_query_fallback("without people", history),
            "red roses without people",
        )
        self.assertEqual(
            build_contextual_query_fallback("blue ones", history),
            "blue roses",
        )
        self.assertEqual(
            build_contextual_query_fallback("best matches", history),
            "red roses",
        )
        self.assertEqual(
            build_contextual_query_fallback("blue sky", history),
            "blue sky",
        )

    def test_followup_resolver_failure_uses_contextual_fallback(self):
        agent = object.__new__(AgentSquad)
        agent.llm = FailingResolverLlm()
        history = [
            {"role": "user", "content": "red roses"},
            {"role": "assistant", "content": "I found red rose images."},
        ]

        resolved, source = agent._resolve_followup_query_with_source(
            "without people",
            history,
        )

        self.assertEqual(resolved, "red roses without people")
        self.assertEqual(source, "contextual_fallback")

    def test_followup_resolver_rewrites_weak_anaphoric_output(self):
        agent = object.__new__(AgentSquad)
        agent.llm = WeakAnaphoraResolverLlm()
        history = [
            {"role": "user", "content": "red roses"},
            {"role": "assistant", "content": "I found red rose images."},
        ]

        resolved, source = agent._resolve_followup_query_with_source(
            "blue ones",
            history,
        )

        self.assertEqual(resolved, "blue roses")
        self.assertEqual(source, "contextual_anaphora")

    def test_text_intent_fast_path_drops_placeholder_nouns(self):
        result = detect_text_query_intent(
            "blue ones",
            ExplodingLlm(),
            DummyCategoryFilter(),
        )

        self.assertEqual(result.entity_terms, ["blue"])
        self.assertEqual(result.boolean_query, "blue")

    def test_text_intent_llm_prompt_includes_prior_user_context(self):
        llm = JsonIntentLlm()
        history = [
            {"role": "user", "content": "red roses"},
            {"role": "assistant", "content": "I found red rose images."},
        ]

        with patch.object(settings, "text_query_intent_llm_enabled", True):
            result = detect_text_query_intent(
                "red roses without people",
                llm,
                DummyCategoryFilter(),
                conversation_history=history,
                latest_user_query="without people",
            )

        human_prompt = llm.messages[1].content
        self.assertIn("Conversation context", human_prompt)
        self.assertIn("- red roses", human_prompt)
        self.assertIn("Latest user message: without people", human_prompt)
        self.assertIn(
            "Context-aware search query to analyze: red roses without people",
            human_prompt,
        )
        self.assertEqual(result.boolean_query, "red AND roses")
        self.assertEqual(result.exclusion_terms, ["people"])

    def test_direct_hybrid_workflow_step_records_embedding_model(self):
        agent = object.__new__(AgentSquad)
        agent.llm_model = "meta/llama-3.3-70b-instruct"
        steps = []
        embedding_metadata = {
            "provider": "nvidia",
            "model": "nvidia/llama-nemotron-embed-1b-v2",
            "dimensions": 384,
            "send_dimensions": True,
            "query_input_type": "query",
            "passage_input_type": "passage",
            "truncate": "END",
            "vector_field": "dense_vector_nvidia_384",
            "timeout_seconds": 60.0,
        }
        search_result = {
            "results": [{"hadron_id": "1"}],
            "total": 1,
            "took_ms": 42,
            "opensearch_query": {
                "query": {
                    "hybrid": {
                        "queries": [
                            {"knn": {"dense_vector_nvidia_384": {"vector": [0.1] * 384}}},
                            {"bool": {"must": []}},
                        ]
                    }
                }
            },
            "opensearch_index": "icc_images_ext",
            "opensearch_pipeline": "reveal-hybrid",
            "embedding_metadata": embedding_metadata,
        }

        with patch(
            "app.services.agent_squad.photo_search_service.execute_direct_hybrid_search",
            return_value=search_result,
        ):
            result = agent._execute_direct_image_hybrid_search(
                steps=steps,
                workflow_label="Text-Only",
                semantic_query="red roses",
                lexical_query="red roses",
                category_gids=[],
                exclusion_terms=[],
                refinement_filters=[],
                show_generated=False,
                is_not_generated=False,
                pipeline="image",
                search_mode="relevance",
                lexical_operator="and",
            )

        self.assertEqual(result, search_result)
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertIn("nvidia/llama-nemotron-embed-1b-v2", step["reasoning"])
        self.assertIn("dense_vector_nvidia_384", step["reasoning"])
        self.assertEqual(step["input"]["embedding"], embedding_metadata)
        self.assertEqual(step["output"]["embedding"], embedding_metadata)


if __name__ == "__main__":
    unittest.main()
