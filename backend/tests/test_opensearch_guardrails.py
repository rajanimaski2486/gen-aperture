import unittest

from app.services.conversation_store import (
    CONVERSATION_INDEX,
    ConversationStore,
    ConversationWriteLimitExceeded,
)
from app.services.opensearch_guardrails import (
    _conversation_request_allowed,
    _readonly_request_allowed,
)


class FakeIndices:
    def __init__(self, *, exists=True, size_bytes=0):
        self._exists = exists
        self._size_bytes = size_bytes

    def exists(self, index):
        return self._exists

    def stats(self, index, metric=None):
        return {
            "indices": {
                index: {
                    "total": {
                        "store": {
                            "size_in_bytes": self._size_bytes,
                        }
                    }
                }
            }
        }


class FakeClient:
    def __init__(self, *, exists=True, count=0, size_bytes=0):
        self.indices = FakeIndices(exists=exists, size_bytes=size_bytes)
        self._count = count

    def count(self, index):
        return {"count": self._count}


def make_store(*, exists=True, count=0, size_bytes=0):
    store = object.__new__(ConversationStore)
    store.index = CONVERSATION_INDEX
    store.client = FakeClient(exists=exists, count=count, size_bytes=size_bytes)
    return store


class OpenSearchGuardrailTests(unittest.TestCase):
    def test_readonly_guardrails_still_allow_only_search_posts(self):
        self.assertTrue(_readonly_request_allowed("GET", "/web-index-v9/_doc/1"))
        self.assertTrue(_readonly_request_allowed("POST", "/web-index-v9/_search"))
        self.assertFalse(_readonly_request_allowed("POST", "/web-index-v9/_bulk"))
        self.assertFalse(_readonly_request_allowed("PUT", "/web-index-v9"))

    def test_conversation_guardrails_allow_only_conversation_index_writes(self):
        idx = CONVERSATION_INDEX
        self.assertTrue(_conversation_request_allowed("PUT", f"/{idx}", idx))
        self.assertTrue(_conversation_request_allowed("PUT", f"/{idx}/_doc/1", idx))
        self.assertTrue(_conversation_request_allowed("POST", f"/{idx}/_update/1", idx))
        self.assertTrue(_conversation_request_allowed("DELETE", f"/{idx}/_doc/1", idx))

        self.assertFalse(_conversation_request_allowed("PUT", "/web-index-v9", idx))
        self.assertFalse(_conversation_request_allowed("POST", "/_bulk", idx))
        self.assertFalse(_conversation_request_allowed("DELETE", f"/{idx}", idx))
        self.assertFalse(_conversation_request_allowed("PUT", "/_plugins/_ism/policies/p", idx))


class ConversationLimitTests(unittest.TestCase):
    def test_new_record_is_rejected_at_record_limit(self):
        store = make_store(count=5000)
        with self.assertRaises(ConversationWriteLimitExceeded):
            store._assert_write_allowed(new_record=True)

    def test_existing_record_update_is_allowed_at_record_limit(self):
        store = make_store(count=5000)
        store._assert_write_allowed(new_record=False)

    def test_existing_record_update_is_rejected_above_record_limit(self):
        store = make_store(count=5001)
        with self.assertRaises(ConversationWriteLimitExceeded):
            store._assert_write_allowed(new_record=False)

    def test_write_is_rejected_at_store_size_limit(self):
        store = make_store(size_bytes=5 * 1024 * 1024 * 1024)
        with self.assertRaises(ConversationWriteLimitExceeded):
            store._assert_write_allowed()

    def test_projected_write_is_rejected_when_it_would_exceed_size_limit(self):
        max_size = 5 * 1024 * 1024 * 1024
        store = make_store(size_bytes=max_size - 10)
        with self.assertRaises(ConversationWriteLimitExceeded):
            store._assert_write_allowed(projected_bytes=11)

    def test_missing_index_does_not_allow_implicit_write_creation(self):
        store = make_store(exists=False)
        with self.assertRaises(ConversationWriteLimitExceeded):
            store._assert_write_allowed(new_record=True)


if __name__ == "__main__":
    unittest.main()
