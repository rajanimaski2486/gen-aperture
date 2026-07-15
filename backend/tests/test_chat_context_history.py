import json
import unittest

from app.routers.chat import (
    MAX_CLIENT_HISTORY_CONTENT_CHARS,
    MAX_CLIENT_HISTORY_MESSAGES,
    _normalize_client_conversation_history,
    _select_conversation_history_for_agent,
)


class ChatContextHistoryTests(unittest.TestCase):
    def test_normalize_client_history_keeps_recent_chat_messages(self):
        raw = json.dumps(
            [
                {"role": "system", "content": "ignore me"},
                {"role": "user", "content": "  red   roses  "},
                {"role": "assistant", "content": "I found red rose images."},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "x" * (MAX_CLIENT_HISTORY_CONTENT_CHARS + 25)},
            ]
        )

        result = _normalize_client_conversation_history(raw)

        self.assertEqual(result[0], {"role": "user", "content": "red roses"})
        self.assertEqual(
            result[1],
            {"role": "assistant", "content": "I found red rose images."},
        )
        self.assertEqual(result[2]["role"], "assistant")
        self.assertEqual(len(result[2]["content"]), MAX_CLIENT_HISTORY_CONTENT_CHARS)

    def test_normalize_client_history_limits_total_messages(self):
        raw = json.dumps(
            [
                {"role": "user", "content": f"message {idx}"}
                for idx in range(MAX_CLIENT_HISTORY_MESSAGES + 5)
            ]
        )

        result = _normalize_client_conversation_history(raw)

        self.assertEqual(len(result), MAX_CLIENT_HISTORY_MESSAGES)
        self.assertEqual(result[0]["content"], "message 5")

    def test_malformed_client_history_is_ignored(self):
        self.assertEqual(_normalize_client_conversation_history("{not-json"), [])
        self.assertEqual(_normalize_client_conversation_history(json.dumps({"role": "user"})), [])

    def test_client_history_wins_over_stale_server_history(self):
        server_history = [{"role": "user", "content": "old server message"}]
        client_history = [{"role": "user", "content": "fresh UI message"}]

        self.assertEqual(
            _select_conversation_history_for_agent(server_history, client_history),
            client_history,
        )
        self.assertEqual(
            _select_conversation_history_for_agent(server_history, []),
            server_history,
        )


if __name__ == "__main__":
    unittest.main()
