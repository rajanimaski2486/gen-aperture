"""
Reflection-Style Reranker for the Gen-Aperture search pipeline.

Architecture overview
─────────────────────
Given the user query and a list of OpenSearch-retrieved photo candidates, the
reranker runs THREE passes:

  Pass 1 – LLM Scoring      : Batch-score all candidates against the query
  Pass 2 – LLM Critique     : Identify duplicates, borderline cases, and validate ranking
  Pass 3 – Python Selection : Deterministic filtering, deduplication, and final ranking

The reranker is ONLY activated when the user's message contains a known trigger
phrase (e.g. "best", "top ranked", "rerank", "reflect and respond", "reviewed").

Example trigger queries
───────────────────────
  "Show me the best ocean sunset photos"      → triggers
  "Top ranked nature images for my project"  → triggers
  "Reflect and respond with reviewed picks"  → triggers
  "Find a photo of a dog"                    → does NOT trigger
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


def _run_async_from_sync(coro: Any) -> Any:
    """
    Execute an async coroutine from a synchronous call frame, even when an
    asyncio event loop is already running in the current thread (e.g. when a
    sync LangGraph node is called from an async FastAPI endpoint).

    Strategy
    ────────
    • If NO loop is running  → call asyncio.run() directly (simple path).
    • If a loop IS running   → submit the coroutine to a *new* loop running
      inside a ThreadPoolExecutor worker thread, then block until done.

    The thread-pool approach avoids the ``RuntimeError: This event loop is
    already running`` that ``loop.run_until_complete()`` raises when called
    from within a running loop.
    """
    try:
        asyncio.get_running_loop()
        # Event loop already running — dispatch to a worker thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop
        return asyncio.run(coro)

# ─────────────────────────────────────────────────────────────────────────────
# Trigger-pattern detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that activate the reranker. Case-insensitive re.search is used.
# Covers: "best", "best results", "best matching", "top ranked", "top rank",
#         "rerank", "reflect and respond", "reviewed"
_RERANK_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bbest\b", re.IGNORECASE),
    re.compile(r"\btop[-\s]?ranked?\b", re.IGNORECASE),
    re.compile(r"\brerank\b", re.IGNORECASE),
    re.compile(r"\breflect\s+and\s+respond\b", re.IGNORECASE),
    re.compile(r"\breviewed\b", re.IGNORECASE),
]


def should_rerank(user_message: str) -> bool:
    """Return True if the user's message contains a reranking trigger phrase."""
    return any(p.search(user_message) for p in _RERANK_TRIGGER_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RerankerConfig:
    """
    Thresholds and limits that govern the reranker's behaviour.

    All raw_score values are on a 0-10 scale (set by the LLM in Pass 1).
    rerank_score in the final output is normalised to 0-1.
    """

    # Minimum number of results the reranker should include in the final set
    min_results_target: int = 10

    # raw_score (0-10) below which a result is a poor match
    relevance_threshold: float = 5.0

    # raw_score in [borderline_threshold, relevance_threshold) may be promoted
    # if needed to reach min_results_target
    borderline_threshold: float = 3.5

    # Jaccard keyword-overlap above which two results are treated as near-duplicates
    duplicate_similarity_threshold: float = 0.5

    # Maximum number of candidates fed to Pass 2 (critique phase)
    # Using all 50 in the critique prompt would be too noisy; keep top-25
    max_candidates_for_critique: int = 25

    # Model used for both LLM passes
    model: str = "Qwen3-VL-Reranker-8B"

    @classmethod
    def from_settings(cls, settings: Any) -> "RerankerConfig":
        """Build from the application Settings object."""
        return cls(
            min_results_target=settings.rerank_min_results_target,
            relevance_threshold=settings.rerank_relevance_threshold,
            borderline_threshold=settings.rerank_borderline_threshold,
            duplicate_similarity_threshold=settings.rerank_duplicate_similarity_threshold,
            model=settings.rerank_model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedDecision:
    """
    The reranker's verdict on a single candidate result.

    Mirrors the RerankerDecision Pydantic schema used by the API layer so
    that fields can be serialised with dataclasses.asdict() before being
    stored in AgentState.
    """
    hadron_id: str | None = None
    ext_id: Any = None
    final_rank: int | None = None      # 1-based rank; None if discarded
    rerank_score: float = 0.0          # Normalised 0-1
    keep: bool = True
    is_borderline: bool = False        # Promoted only to reach min_results_target
    reason: str = ""
    matched_criteria: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RerankerOutput:
    """
    Aggregated output from a single reranker run.

    ``triggered`` is False when reranking was skipped due to an error —
    the caller should treat this the same as if the reranker was never invoked.
    """
    triggered: bool
    total_candidates: int
    final_results: list[dict[str, Any]]        # Filtered, reranked raw result dicts
    decisions: list[RankedDecision]            # One per original candidate (kept + discarded)
    explanation: str | None                    # Set when fewer than min_results_target were kept
    pass_summaries: dict[str, Any]             # Debugging info from each pass


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_scores(scores: list[float]) -> list[float]:
    """
    Min-max normalise a list of scores to [0, 1].

    If all scores are identical (or the list has one element) every value
    is mapped to 1.0 so that the results are still usable.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _compute_keyword_jaccard(a: list[str], b: list[str]) -> float:
    """
    Return the Jaccard similarity between two keyword lists.

    Used as a lightweight, Python-only duplicate-detection fallback when the
    LLM critique does not explicitly identify duplicate groups.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    set_a = {kw.lower().strip() for kw in a}
    set_b = {kw.lower().strip() for kw in b}
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _build_candidate_summary(candidates: list[dict[str, Any]], *, max_keywords: int = 30) -> str:
    """
    Build a compact JSON-like text summary of the candidates for the LLM prompts.

    Includes all fields the LLM needs to make an accurate relevance judgement:
    hadron_id, description, keywords (up to max_keywords), category_ids,
    license_count, and the raw retrieval score.
    """
    rows = []
    for i, c in enumerate(candidates, 1):
        kws = c.get("keywords", [])
        rows.append({
            "index": i,
            "hadron_id": c.get("hadron_id") or f"__noid_{id(c)}",
            "description": (c.get("description") or "")[:300],
            "keywords": kws[:max_keywords],
            "category_ids": c.get("global_category_ids") or c.get("category_ids") or [],
            "license_count": c.get("license_count", 0),
            "retrieval_score": round(c.get("score", 0.0), 4),
        })
    return json.dumps(rows, indent=2)


def _python_dedup(
    candidates_by_id: dict[str, dict[str, Any]],
    threshold: float,
) -> list[str]:
    """
    Python-side near-duplicate detection via Jaccard keyword similarity.

    For each pair of candidates we compute the keyword Jaccard.  When it
    exceeds ``threshold`` we keep the one with the higher retrieval score and
    flag the other as a duplicate.  Returns the list of hadron_ids to DISCARD.
    """
    ids = list(candidates_by_id.keys())
    discard: set[str] = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if ids[i] in discard or ids[j] in discard:
                continue
            a = candidates_by_id[ids[i]]
            b = candidates_by_id[ids[j]]
            sim = _compute_keyword_jaccard(
                a.get("keywords", []), b.get("keywords", [])
            )
            if sim >= threshold:
                # Discard the one with the lower retrieval score
                if a.get("score", 0) >= b.get("score", 0):
                    discard.add(ids[j])
                else:
                    discard.add(ids[i])
    return list(discard)


# ─────────────────────────────────────────────────────────────────────────────
# Main reranker class
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionReranker:
    """
    A reflection-style post-retrieval reranker.

    Usage
    ─────
        config = RerankerConfig.from_settings(settings)
        reranker = ReflectionReranker(config)

        if reranker.should_rerank(user_message):
            output = await reranker.rerank(
                user_query=user_message,
                candidates=raw_search_results,
                search_criteria={"requirements": ..., "refinement_filters": ...},
            )
            if output.triggered:
                final_results = output.final_results
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config

    # ── public interface ──────────────────────────────────────────────────────

    @staticmethod
    def should_rerank(user_message: str) -> bool:
        """Delegate to module-level helper for easy testability."""
        return should_rerank(user_message)

    async def rerank(
        self,
        user_query: str,
        candidates: list[dict[str, Any]],
        search_criteria: dict[str, Any] | None = None,
    ) -> RerankerOutput:
        """
        Orchestrate the three-pass reflection reranking pipeline.

        Parameters
        ──────────
        user_query       The raw user message / search query.
        candidates       Top-k results returned by OpenSearch (raw dicts from
                         photo_search.execute_raw_query).
        search_criteria  Optional structured context extracted upstream
                         (requirements, filters, exclusions, category_gids).

        Returns a RerankerOutput.  If any unrecoverable error occurs during
        the LLM passes, ``triggered=False`` is returned so the caller can
        fall back to the original candidate list.
        """
        if not candidates:
            return RerankerOutput(
                triggered=False,
                total_candidates=0,
                final_results=[],
                decisions=[],
                explanation="No candidates to rerank.",
                pass_summaries={},
            )

        total = len(candidates)
        criteria_text = _format_criteria(search_criteria)

        try:
            client = AsyncOpenAI()  # uses OPENAI_API_KEY from environment

            # ── Pass 1: Scoring ───────────────────────────────────────────────
            logger.info("Reranker Pass 1 (scoring): evaluating %d candidates", total)
            pass1_result = await self._scoring_pass(client, user_query, criteria_text, candidates)

            # ── Pass 2: Critique ──────────────────────────────────────────────
            top_n = pass1_result[: self.config.max_candidates_for_critique]
            logger.info("Reranker Pass 2 (critique): reviewing top %d candidates", len(top_n))
            pass2_result = await self._critique_pass(client, user_query, criteria_text, top_n, candidates)

            # ── Pass 3: Final selection (pure Python) ─────────────────────────
            logger.info("Reranker Pass 3 (selection): building final ranked list")
            output = self._final_selection(candidates, pass1_result, pass2_result)
            return output

        except Exception as exc:
            # Any failure falls back gracefully — original order is preserved
            logger.warning(
                "Reranker failed, returning original results: %s", exc, exc_info=True
            )
            return RerankerOutput(
                triggered=False,
                total_candidates=total,
                final_results=candidates,
                decisions=[],
                explanation=None,
                pass_summaries={"error": str(exc)},
            )

    # ── Pass 1: LLM Scoring ───────────────────────────────────────────────────

    async def _scoring_pass(
        self,
        client: AsyncOpenAI,
        user_query: str,
        criteria_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Ask the LLM to score each candidate on four dimensions then combine
        them into a single raw_score (0-10 scale).

        Scoring rubric applied inside the prompt:
            raw_score = query_relevance * 0.40
                      + criteria_match  * 0.25
                      + specificity     * 0.25
                      + completeness    * 0.10

        Returns a list of scoring dicts, sorted by raw_score descending.
        Each dict has the keys: hadron_id, query_relevance, criteria_match,
        specificity, completeness, raw_score, notes.
        """
        candidate_text = _build_candidate_summary(candidates)

        system_prompt = (
            "You are an expert stock photo search quality evaluator. "
            "Score each candidate photo against the user's search query using only the "
            "photo's description and keywords — not its retrieval rank.\n\n"
            "Scoring dimensions (all 0-10). Use the full 0-10 range:\n"
            "  query_relevance:\n"
            "    10 = photo is literally and specifically OF the queried subject/scene\n"
            "     7 = photo clearly depicts the main subject but misses one detail\n"
            "     5 = loosely related — shares a theme or setting but not the subject\n"
            "     2 = tangentially related at best\n"
            "     0 = completely off-topic\n"
            "  criteria_match:\n"
            "    10 = satisfies every explicit constraint (location, season, style, exclusions)\n"
            "    10 = if NO explicit constraints exist, score same as query_relevance\n"
            "     5 = satisfies most constraints but misses one\n"
            "     0 = ONLY if a hard constraint is explicitly violated (wrong location, excluded keyword present)\n"
            "  specificity:\n"
            "    10 = photo is unmistakably about the specific subject requested\n"
            "     5 = could plausibly match several different queries\n"
            "     2 = so generic it adds no value\n"
            "  completeness:\n"
            "    10 = rich, detailed scene with strong visual substance\n"
            "     5 = adequate but sparse\n"
            "     2 = empty, abstract, or visually uninformative\n\n"
            "Combined score formula:\n"
            "  raw_score = query_relevance*0.50 + criteria_match*0.25 + specificity*0.20 + completeness*0.05\n\n"
            "IMPORTANT: These results were already ranked as relevant by a production search engine. "
            "Give benefit of the doubt — only score below 4 when the photo is clearly "
            "off-topic or violates an explicit constraint. A photo that generally matches the "
            "subject should score 6+. Do NOT over-penalise based on incomplete descriptions.\n\n"
            "Return ONLY valid JSON — a top-level object with a single key 'scores' "
            "whose value is an array. Each array element must have exactly these keys:\n"
            "  hadron_id, query_relevance, criteria_match, specificity, completeness, raw_score, notes\n\n"
            "The 'notes' field must explain in one sentence WHY this score was given, "
            "referencing specific description/keyword evidence. "
            "Do not include any text outside the JSON object."
        )

        user_prompt = (
            f"## User Search Query\n\"{user_query}\"\n\n"
            f"## Search Criteria & Constraints\n{criteria_text}\n\n"
            f"## Candidate Photos\n"
            f"Each entry includes: hadron_id, description, keywords, category_ids, "
            f"license_count, retrieval_score.\n\n"
            f"{candidate_text}\n\n"
            "Score every candidate against the search query. "
            "Return JSON as described."
        )

        response = await client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        scores: list[dict[str, Any]] = data.get("scores", [])

        # Sort by raw_score descending so Pass 2 sees the most relevant first
        scores.sort(key=lambda x: x.get("raw_score", 0), reverse=True)
        return scores

    # ── Pass 2: LLM Critique ──────────────────────────────────────────────────

    async def _critique_pass(
        self,
        client: AsyncOpenAI,
        user_query: str,
        criteria_text: str,
        top_scored: list[dict[str, Any]],
        all_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Ask the LLM to critically review the top-scored results.

        The critique agent is asked four reflection questions:
          1. "Does this result actually satisfy the search intent?"
          2. "Is this only keyword overlap, or a true match?"
          3. "Would I defend including this in the top 10?"
          4. "Is there a stronger result that makes this redundant?"

        Returns a dict with keys:
          duplicate_groups:    [[id_to_keep, id_dup1, id_dup2], ...]
          confident_includes:  [hadron_id, ...]
          borderline_ids:      [hadron_id, ...]
          confident_excludes:  [hadron_id, ...]
          final_ranked_ids:    [hadron_id, ...]  (ordered best-first)
          challenges:          {hadron_id: "reason the ranking was challenged"}
        """
        # Build a rich critique view: Pass-1 scores PLUS full description + keywords
        # so the LLM can actually re-evaluate content, not just trust the scores.
        candidates_by_id_local: dict[str, dict[str, Any]] = {
            c.get("hadron_id") or f"__noid_{id(c)}": c
            for c in all_candidates
        }
        critique_rows = []
        for s in top_scored:
            hid = s.get("hadron_id")
            c = candidates_by_id_local.get(hid or "", {})
            kws = c.get("keywords", [])
            critique_rows.append({
                "hadron_id": hid,
                "pass1_raw_score": round(s.get("raw_score", 0), 2),
                "pass1_notes": s.get("notes", ""),
                "description": (c.get("description") or "")[:300],
                "keywords": kws[:30],
                "category_ids": c.get("global_category_ids") or c.get("category_ids") or [],
            })
        critique_summary = json.dumps(critique_rows, indent=2)

        system_prompt = (
            "You are a senior stock photo search quality reviewer. "
            "You are given an initial scoring pass and the actual photo descriptions/keywords. "
            "Your job is to improve result ordering and remove only clearly irrelevant photos.\n\n"
            "For each candidate, read the description and keywords carefully, then ask:\n"
            "  1. Does this photo broadly match what the user searched for? "
            "A partial match or thematically related photo is fine to keep.\n"
            "  2. Only exclude when: the query requires a SPECIFIC location/landmark/season "
            "and the photo clearly lacks it (e.g. query='empire state building', "
            "photo has no NYC/skyscraper context).\n"
            "  3. Are any two results nearly identical (same subject, similar composition)?\n\n"
            "Default to KEEPING results. Only put a result in confident_excludes if it is "
            "clearly off-topic. Borderline or imperfect matches belong in borderline_ids, "
            "not confident_excludes.\n\n"
            "Return ONLY valid JSON with exactly these top-level keys:\n"
            "  duplicate_groups:   array of arrays; each inner array lists hadron_ids of "
            "near-duplicates. The first id in each inner array is the one to KEEP.\n"
            "  confident_includes: hadron_ids you are confident are good matches.\n"
            "  borderline_ids:     hadron_ids that partially match but are not ideal.\n"
            "  confident_excludes: hadron_ids that are clearly off-topic or violate a hard constraint.\n"
            "  final_ranked_ids:   all remaining hadron_ids ordered best-first "
            "(omit ids in confident_excludes).\n"
            "  challenges:         object mapping hadron_id → one-sentence reason if you "
            "challenge the Pass-1 score (leave empty {} if no challenges).\n\n"
            "Do not include any text outside the JSON object."
        )

        user_prompt = (
            f"## User Search Query\n\"{user_query}\"\n\n"
            f"## Search Criteria & Constraints\n{criteria_text}\n\n"
            f"## Top-Scored Candidates (Pass 1 scores + full photo data)\n"
            f"{critique_summary}\n\n"
            "Review each candidate against the search query and photo content. "
            "Return your critique as JSON as described."
        )

        response = await client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    # ── Pass 3: Final selection (pure Python) ─────────────────────────────────

    def _final_selection(
        self,
        candidates: list[dict[str, Any]],
        pass1_scores: list[dict[str, Any]],
        pass2_critique: dict[str, Any],
    ) -> RerankerOutput:
        """
        Deterministic Python pass that converts Pass 1 scores + Pass 2 critique
        into the final reranked result list and a full decision log.

        Steps
        ─────
        1. Build lookup maps (hadron_id → candidate dict, hadron_id → score dict)
        2. Run Python-side Jaccard dedup as a safety net for any duplicates the
           LLM critique missed
        3. Remove ``confident_excludes`` (LLM-flagged) and Jaccard duplicates
        4. Normalise raw_scores to [0, 1] across the surviving set
        5. Filter results below relevance_threshold (using un-normalised raw_score)
        6. If below min_results_target, promote borderline_ids in LLM-ranked order
        7. Build final_ranked_ids ordering from Pass 2 (supplement with remaining
           survivors if the critique did not mention all of them)
        8. Re-assign 1-based final_rank
        9. Emit one RankedDecision per original candidate (kept + discarded)
        10. Set explanation if fewer than min_results_target were kept
        """
        cfg = self.config

        # ── 1. Build lookup maps ──────────────────────────────────────────────
        candidates_by_id: dict[str, dict[str, Any]] = {}
        for c in candidates:
            hid = c.get("hadron_id") or f"__noid_{id(c)}"
            candidates_by_id[hid] = c

        scores_by_id: dict[str, dict[str, Any]] = {}
        for s in pass1_scores:
            hid = s.get("hadron_id")
            if hid:
                scores_by_id[hid] = s

        # For candidates that were never scored (Pass 1 might truncate on error),
        # synthesise a minimal scoring dict using retrieval score alone
        for hid, c in candidates_by_id.items():
            if hid not in scores_by_id:
                raw = c.get("score", 0.0) * 10  # scale OpenSearch score to 0-10 approx
                scores_by_id[hid] = {
                    "hadron_id": hid,
                    "raw_score": raw,
                    "query_relevance": raw,
                    "criteria_match": raw,
                    "specificity": raw,
                    "completeness": raw,
                    "notes": "Score estimated from retrieval score (not evaluated by LLM)",
                }

        # ── 2. Collect explicit removes from critique ─────────────────────────
        confident_excludes: set[str] = set(pass2_critique.get("confident_excludes", []))
        borderline_ids: list[str] = pass2_critique.get("borderline_ids", [])
        final_ranked_ids_from_llm: list[str] = pass2_critique.get("final_ranked_ids", [])

        # ── 3. Python-side Jaccard dedup ──────────────────────────────────────
        # Only run on survivors so we don't accidentally re-discard already excluded items
        survivors_for_dedup = {
            hid: c for hid, c in candidates_by_id.items() if hid not in confident_excludes
        }
        jaccard_duplicates = set(
            _python_dedup(survivors_for_dedup, cfg.duplicate_similarity_threshold)
        )

        # Combine both exclude sets
        all_excludes: set[str] = confident_excludes | jaccard_duplicates

        # ── 4. Surviving candidates pool ──────────────────────────────────────
        surviving_ids = [hid for hid in candidates_by_id if hid not in all_excludes]

        # ── 5. Normalise scores across ALL candidates (not just survivors) ────
        # This ensures excluded items display their actual relative score (not 0).
        all_ids = list(candidates_by_id.keys())
        all_raw_scores = [scores_by_id[hid].get("raw_score", 0.0) for hid in all_ids]
        all_norm_scores = _normalize_scores(all_raw_scores)
        norm_by_id: dict[str, float] = dict(zip(all_ids, all_norm_scores))
        raw_by_id: dict[str, float] = dict(zip(all_ids, all_raw_scores))

        # ── 6. Apply relevance threshold ──────────────────────────────────────
        # Use the un-normalised raw_score (0-10 scale) to avoid threshold confusion
        below_threshold: set[str] = {
            hid for hid in surviving_ids
            if raw_by_id[hid] < cfg.relevance_threshold
            and hid not in borderline_ids
        }
        kept_ids = [hid for hid in surviving_ids if hid not in below_threshold]
        discarded_ids = list(below_threshold | all_excludes)

        # ── 7. Build ordered final ranking ────────────────────────────────────
        # Start with the LLM-proposed order (only include ids still in kept_ids)
        llm_ordered = [hid for hid in final_ranked_ids_from_llm if hid in set(kept_ids)]
        # Append any kept_ids not mentioned by the LLM, sorted by raw_score desc
        mentioned = set(llm_ordered)
        extra = sorted(
            [hid for hid in kept_ids if hid not in mentioned],
            key=lambda hid: raw_by_id.get(hid, 0),
            reverse=True,
        )
        ordered_kept = llm_ordered + extra

        # ── 8. Promote borderline candidates if below min_results_target ──────
        promoted_borderline: set[str] = set()
        if len(ordered_kept) < cfg.min_results_target:
            for hid in borderline_ids:
                if len(ordered_kept) >= cfg.min_results_target:
                    break
                if hid in candidates_by_id and hid not in all_excludes and hid not in set(ordered_kept):
                    ordered_kept.append(hid)
                    promoted_borderline.add(hid)

        # ── 8b. Last-resort fallback: if still below target, promote best items
        #        from below_threshold (sorted by raw_score desc) so we never
        #        return 0 results when candidates exist.
        if len(ordered_kept) < cfg.min_results_target:
            fallback_pool = sorted(
                [hid for hid in below_threshold if hid not in all_excludes and hid not in set(ordered_kept)],
                key=lambda hid: raw_by_id.get(hid, 0.0),
                reverse=True,
            )
            for hid in fallback_pool:
                if len(ordered_kept) >= cfg.min_results_target:
                    break
                ordered_kept.append(hid)
                promoted_borderline.add(hid)

        # ── 9. Set explanation if still below target ──────────────────────────
        explanation: str | None = None
        if len(ordered_kept) < cfg.min_results_target:
            explanation = (
                f"Only {len(ordered_kept)} high-quality match"
                f"{'es' if len(ordered_kept) != 1 else ''} found for this query. "
                f"The remaining candidates did not sufficiently satisfy the search criteria."
            )

        # ── 10. Build RankedDecision for every original candidate ─────────────
        all_decisions: list[RankedDecision] = []
        final_id_set = set(ordered_kept)
        kept_set = set(ordered_kept)

        for rank_idx, hid in enumerate(ordered_kept, start=1):
            c = candidates_by_id.get(hid, {})
            s = scores_by_id.get(hid, {})
            norm = norm_by_id.get(hid, 0.0)
            is_borderline_flag = hid in promoted_borderline

            # Build matched_criteria list from critique
            matched: list[str] = []
            if hid in pass2_critique.get("confident_includes", []):
                matched.append("confirmed_include")
            if is_borderline_flag:
                matched.append("promoted_borderline")

            reason = ""
            if hid in (pass2_critique.get("challenges") or {}):
                reason = pass2_critique["challenges"][hid]
            elif s.get("notes"):
                reason = s["notes"]

            confidence = norm  # use normalised score as proxy for confidence
            if is_borderline_flag:
                confidence = min(confidence, 0.55)  # cap borderline confidence

            all_decisions.append(RankedDecision(
                hadron_id=hid,
                ext_id=c.get("ext_id"),
                final_rank=rank_idx,
                rerank_score=round(norm, 4),
                keep=True,
                is_borderline=is_borderline_flag,
                reason=reason,
                matched_criteria=matched,
                confidence=round(confidence, 4),
            ))

        # Discarded candidates — the union of all_excludes and below_threshold
        all_discarded_ids = (all_excludes | below_threshold) - final_id_set
        for hid in all_discarded_ids:
            if hid not in candidates_by_id:
                continue
            c = candidates_by_id[hid]
            s = scores_by_id.get(hid, {})
            norm = norm_by_id.get(hid, 0.0)

            if hid in confident_excludes:
                reason_text = "Excluded by reflection critique: not relevant enough."
            elif hid in jaccard_duplicates:
                reason_text = "Near-duplicate of a higher-ranked result."
            else:
                reason_text = s.get("notes") or "Below relevance threshold."

            all_decisions.append(RankedDecision(
                hadron_id=hid,
                ext_id=c.get("ext_id"),
                final_rank=None,
                rerank_score=round(norm, 4),
                keep=False,
                is_borderline=False,
                reason=reason_text,
                matched_criteria=[],
                confidence=round(norm, 4),
            ))

        # ── 11. Build final result list ───────────────────────────────────────
        final_results = [candidates_by_id[hid] for hid in ordered_kept if hid in candidates_by_id]

        return RerankerOutput(
            triggered=True,
            total_candidates=len(candidates),
            final_results=final_results,
            decisions=all_decisions,
            explanation=explanation,
            pass_summaries={
                "pass1_scored_count": len(pass1_scores),
                "pass2_confident_includes": len(pass2_critique.get("confident_includes", [])),
                "pass2_confident_excludes": len(confident_excludes),
                "pass2_borderline_count": len(borderline_ids),
                "jaccard_duplicates_removed": len(jaccard_duplicates),
                "below_threshold_removed": len(below_threshold),
                "promoted_borderline": len(promoted_borderline),
                "final_kept": len(ordered_kept),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_criteria(search_criteria: dict[str, Any] | None) -> str:
    """
    Convert the upstream search_criteria dict into a human-readable text block
    for inclusion in LLM prompts.

    Expected keys (all optional):
      user_query:         the resolved search topic (injected by callers for text-only searches)
      requirements:       dict from Project Manager (brand domain, visual constraints, etc.)
      refinement_filters: list of OpenSearch filter clauses
      exclusion_terms:    list of keyword exclusion strings
      category_gids:      list of category GIDs
    """
    if not search_criteria:
        return "No explicit search criteria provided."

    lines: list[str] = []

    # Primary criterion: the actual search subject — always include when available
    user_query = search_criteria.get("user_query")
    if user_query:
        lines.append(f"Primary search subject: \"{user_query}\"")
        lines.append(
            "Photos that depict or are thematically related to this subject are relevant. "
            "Only exclude results that are clearly unrelated."
        )

    requirements = search_criteria.get("requirements")
    if requirements and isinstance(requirements, dict):
        if requirements.get("brand_domain"):
            lines.append(f"Brand domain: {requirements['brand_domain']}")
        if requirements.get("visual_requirements"):
            lines.append(f"Visual requirements: {', '.join(requirements['visual_requirements'])}")
        if requirements.get("technical_requirements"):
            lines.append(f"Technical requirements: {', '.join(requirements['technical_requirements'])}")
        if requirements.get("themes_moods"):
            moods = requirements["themes_moods"]
            if isinstance(moods, list):
                lines.append(f"Mood/themes: {', '.join(moods)}")
            else:
                lines.append(f"Mood/themes: {moods}")
        if requirements.get("mood_tone"):
            lines.append(f"Mood/tone: {requirements['mood_tone']}")

    exclusions = search_criteria.get("exclusion_terms") or []
    if exclusions:
        lines.append(f"Exclude photos containing these keywords: {', '.join(exclusions)}")

    filters = search_criteria.get("refinement_filters") or []
    if filters:
        filter_descriptions = []
        for f in filters:
            if isinstance(f, dict):
                # Handle term filters: {"term": {"field": value}}
                if "term" in f:
                    for field, val in f["term"].items():
                        v = val.get("value", val) if isinstance(val, dict) else val
                        filter_descriptions.append(f"{field}={v}")
                # Handle range filters: {"range": {"field": {"gte": ...}}}
                elif "range" in f:
                    for field, bounds in f["range"].items():
                        parts = [f"{k}:{v}" for k, v in bounds.items()]
                        filter_descriptions.append(f"{field} {', '.join(parts)}")
                else:
                    field_name = f.get("field", "")
                    value = f.get("value") or f.get("gte") or ""
                    if field_name:
                        filter_descriptions.append(f"{field_name}={value}")
        if filter_descriptions:
            lines.append(f"Applied filters: {', '.join(filter_descriptions)}")

    category_gids = search_criteria.get("category_gids") or []
    if category_gids:
        lines.append(f"Category GIDs (must match one of): {category_gids}")

    return "\n".join(lines) if lines else "No explicit search criteria provided."



# ─────────────────────────────────────────────────────────────────────────────
# Example usage (for documentation and manual testing only)
# ─────────────────────────────────────────────────────────────────────────────
#
# EXAMPLE INPUT
# ─────────────
# user_query  = "best ocean sunset photos for travel brand"
# candidates  = [
#   {"hadron_id": "h001", "ext_id": 1001, "description": "Golden sunset over calm ocean waters",
#    "keywords": ["sunset", "ocean", "golden", "travel", "sky"],
#    "license_count": 320, "score": 8.5},
#   {"hadron_id": "h002", "ext_id": 1002, "description": "Aerial view of beach at dusk",
#    "keywords": ["beach", "aerial", "dusk", "drone"], "license_count": 180, "score": 7.2},
#   ... (up to 20 candidates)
# ]
# search_criteria = {
#   "requirements": {"brand_domain": "travel", "mood_tone": "warm, inspiring"},
#   "exclusion_terms": ["crowded", "people"],
# }
#
# EXAMPLE OUTPUT
# ──────────────
# RerankerOutput(
#   triggered=True,
#   total_candidates=20,
#   final_results=[...],  # ordered list of 10+ result dicts
#   decisions=[
#     RankedDecision(hadron_id="h001", final_rank=1, rerank_score=0.97, keep=True,
#                    reason="Directly depicts golden ocean sunset; strong travel mood.",
#                    matched_criteria=["confirmed_include"], confidence=0.97),
#     RankedDecision(hadron_id="h002", final_rank=2, rerank_score=0.83, keep=True,
#                    reason="Beach at dusk with warm tones aligns with travel brand.",
#                    matched_criteria=["confirmed_include"], confidence=0.83),
#     RankedDecision(hadron_id="h015", final_rank=None, rerank_score=0.12, keep=False,
#                    reason="Near-duplicate of h001 with weaker composition.",
#                    matched_criteria=[], confidence=0.12),
#     ...
#   ],
#   explanation=None,  # None means >= 10 results were found
#   pass_summaries={
#     "pass1_scored_count": 20,
#     "pass2_confident_includes": 12,
#     "pass2_confident_excludes": 3,
#     "jaccard_duplicates_removed": 2,
#     "below_threshold_removed": 3,
#     "final_kept": 12,
#   }
# )
