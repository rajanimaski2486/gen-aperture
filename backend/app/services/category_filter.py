"""
Category extraction and OpenSearch filter builder for Gen-Aperture.

Loads the category taxonomy from categories.json, matches LLM-extracted
category mentions against valid entries, retrieves their GIDs, and builds
an OpenSearch OR filter across matched GIDs.
"""
import copy
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

# Path to categories resource
_CATEGORIES_FILE = Path(__file__).parent.parent.parent / "resources" / "categories.json"


class CategoryFilter:
    """
    Responsible for:
      1. Loading and indexing the category taxonomy from categories.json.
      2. Matching free-text mentions (from brief analysis) against valid categories.
      3. Building OpenSearch filter clauses for matched GIDs.
    """

    def __init__(self, categories_path: Path = _CATEGORIES_FILE):
        # Normalised label → gid mapping (one entry per normalised token)
        self._label_to_gid: Dict[str, int] = {}
        # gid → canonical display value
        self._gid_to_value: Dict[int, str] = {}
        self._load(categories_path)

    # ------------------------------------------------------------------
    # Loading / indexing
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Load and index categories from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries = data.get("en", [])
            for entry in entries:
                gid = entry.get("gid")
                value = entry.get("value", "")
                namespace = entry.get("namespace", [])

                # Skip deleted / uncategorised sentinel values
                if not value or value in ("DELETED", "NOT-CATEGORIZED"):
                    continue

                # Store canonical display name (prefer local namespace)
                if "local" in namespace or gid not in self._gid_to_value:
                    self._gid_to_value[gid] = value

                # Index the full value string (e.g. "Animals/Wildlife")
                full_key = self._normalize(value)
                if full_key and gid is not None:
                    self._label_to_gid[full_key] = gid

                # Also index individual slash-separated or space-separated parts
                # so "Wildlife" alone can still match gid 1 (Animals/Wildlife)
                parts = re.split(r"[/\s]+", value)
                for part in parts:
                    part_key = self._normalize(part)
                    # Only index meaningful parts (> 3 chars) and don't clobber
                    # a full-value mapping with a shorter partial one
                    if part_key and len(part_key) > 3:
                        self._label_to_gid.setdefault(part_key, gid)

            logger.info(
                "CategoryFilter: Loaded %d unique GIDs, %d label keys",
                len(self._gid_to_value),
                len(self._label_to_gid),
            )
        except Exception as e:
            logger.error("CategoryFilter: Failed to load categories: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_categories(self, text: str) -> List[Dict[str, Any]]:
        """
        Scan *text* for mentions of valid category labels.

        Returns a deduplicated list (one entry per GID), sorted by GID:
            [{"gid": int, "value": str, "matched_term": str}, ...]

        Matching is case-insensitive and uses whole-word boundaries.
        Longer label keys are tried before shorter ones (greedy), so
        "Animals/Wildlife" wins over the partial token "Wildlife".
        """
        normalized = self._normalize(text)
        seen_gids: Set[int] = set()
        matches: List[Dict[str, Any]] = []

        # Iterate longest label first for greedy / most-specific matching
        for label, gid in sorted(
            self._label_to_gid.items(), key=lambda kv: -len(kv[0])
        ):
            if gid in seen_gids:
                continue
            pattern = r"\b" + re.escape(label) + r"\b"
            if re.search(pattern, normalized):
                seen_gids.add(gid)
                matches.append(
                    {
                        "gid": gid,
                        "value": self._gid_to_value.get(gid, label),
                        "matched_term": label,
                    }
                )

        logger.debug("CategoryFilter: Matched %d categories from text", len(matches))
        return sorted(matches, key=lambda m: m["gid"])

    def build_filter(self, gids: List[int]) -> Optional[Dict[str, Any]]:
        """
        Build an OpenSearch ``terms`` filter for ``global_category_ids``
        using OR semantics across all supplied GIDs.

        Returns ``None`` when *gids* is empty.
        """
        if not gids:
            return None
        return {"terms": {"global_category_ids": gids}}

    def inject_into_query(
        self,
        opensearch_query: Dict[str, Any],
        extra_filters: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Deep-copy *opensearch_query* and inject *extra_filters* into the
        ``query.bool.filter`` list.  If the ``bool`` wrapper or ``filter``
        array does not exist it is created.  Returns the modified copy so
        the original is never mutated.
        """
        if not extra_filters:
            return opensearch_query

        q = copy.deepcopy(opensearch_query)
        query_block = q.setdefault("query", {})

        # If the top-level query is not already a ``bool``, wrap it
        if "bool" not in query_block:
            original = copy.deepcopy(query_block)
            query_block.clear()
            query_block["bool"] = {"must": [original]}

        bool_clause = query_block["bool"]

        # Ensure filter is a list
        existing = bool_clause.get("filter", [])
        if not isinstance(existing, list):
            existing = [existing]
        bool_clause["filter"] = existing + extra_filters

        return q

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Lowercase *text*, collapse whitespace, and strip characters that are
        not alphanumeric, spaces, or forward-slashes (retained so that
        "Animals/Wildlife" is kept intact as a compound key).
        """
        text = text.lower().strip()
        # Replace anything that isn't a letter, digit, space or slash with a space
        text = re.sub(r"[^a-z0-9 /]", " ", text)
        # Collapse runs of whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @property
    def all_values(self) -> List[str]:
        """Return all valid category display strings."""
        return list(self._gid_to_value.values())

    def label_for_gid(self, gid: int, default: str = "") -> str:
        """Return the canonical display label for a GID, or *default* if unknown."""
        return self._gid_to_value.get(gid, default or str(gid))


# Module-level singleton — imported by agent_squad and photo_search
category_filter = CategoryFilter()
