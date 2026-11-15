"""Optional Ollama LLM post-processor for karaoke line segmentation and cleanup.

Takes word-level timestamps produced by ASR/alignment and uses an Ollama model
(e.g. ``qwen3:27b``) to group words into natural, singable karaoke lines and
correct obvious homophones/punctuation without altering timestamps.  Any error
or schema mismatch falls back gracefully to deterministic pause-based grouping.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from demucster.models.schemas import LyricsLine, LyricsWord
from demucster.services.lyrics_formats import group_words_into_lines, normalize_words

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a karaoke subtitle editor. Group the given indexed words into natural, "
    "singable karaoke display lines (typically 3 to 9 words and <= 42 characters per line, "
    "breaking at musical pauses and phrasing boundaries). "
    "If a song title or reference lyrics are provided, fix obvious speech-recognition "
    "spelling/homophone errors in the word text without adding, splitting, or removing words. "
    'Respond ONLY with JSON matching: {"lines": [{"word_indices": [0, 1, ...], '
    '"corrected_words": ["...", "..."]}]}. '
    "Every integer index from 0 to N-1 must appear exactly once in strictly ascending order."
)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model output, stripping optional code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama response JSON is not an object")
    return parsed


def _apply_llm_grouping(
    words: list[LyricsWord],
    payload: dict[str, Any],
) -> list[LyricsLine]:
    """Validate LLM line grouping against *words* and build ``LyricsLine`` objects."""
    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("Missing or empty 'lines' array in Ollama response")

    expected_idx = 0
    result: list[LyricsLine] = []

    for entry in raw_lines:
        if not isinstance(entry, dict):
            raise ValueError("Line entry must be an object")
        indices = entry.get("word_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("Line entry missing 'word_indices'")

        corrected = entry.get("corrected_words")
        if corrected is not None and (
            not isinstance(corrected, list) or len(corrected) != len(indices)
        ):
            corrected = None

        line_words: list[LyricsWord] = []
        for pos, raw_idx in enumerate(indices):
            if not isinstance(raw_idx, int) or raw_idx != expected_idx:
                raise ValueError(f"Expected sequential word index {expected_idx}, got {raw_idx!r}")
            if raw_idx >= len(words):
                raise ValueError(f"Word index out of range: {raw_idx}")

            orig = words[raw_idx]
            new_text = orig.word
            if corrected is not None and isinstance(corrected[pos], str):
                candidate = " ".join(corrected[pos].split()).strip()
                if candidate:
                    new_text = candidate

            line_words.append(
                LyricsWord(
                    word=new_text,
                    start=orig.start,
                    end=orig.end,
                    score=orig.score,
                )
            )
            expected_idx += 1

        result.append(
            LyricsLine(
                index=len(result),
                start=line_words[0].start,
                end=line_words[-1].end,
                text=" ".join(w.word for w in line_words),
                words=line_words,
            )
        )

    if expected_idx != len(words):
        raise ValueError(f"Incomplete word coverage: consumed {expected_idx} of {len(words)} words")
    return result


def refine_lines_with_ollama(
    words: list[LyricsWord],
    *,
    ollama_url: str | None,
    model: str = "qwen3:27b",
    title: str | None = None,
    reference_lyrics: str | None = None,
    timeout_seconds: float = 120.0,
    client: httpx.Client | None = None,
) -> tuple[list[LyricsLine], bool]:
    """Group and optionally clean timed *words* using Ollama.

    Returns ``(lines, llm_refined)``.  Falls back to deterministic grouping if
    *ollama_url* is unset, the request fails, or validation rejects the output.
    """
    normalized = normalize_words(words)
    if not normalized:
        return [], False

    if not ollama_url:
        return group_words_into_lines(normalized), False

    compact_words = [
        {"i": idx, "w": w.word, "s": w.start, "e": w.end} for idx, w in enumerate(normalized)
    ]
    user_payload: dict[str, Any] = {
        "word_count": len(compact_words),
        "words": compact_words,
    }
    if title:
        user_payload["title"] = title
    if reference_lyrics:
        user_payload["reference_lyrics"] = reference_lyrics

    request_body = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }

    owns_client = client is None
    http = client or httpx.Client(base_url=ollama_url, timeout=timeout_seconds)
    try:
        response = http.post("/api/chat", json=request_body)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned empty message content")
        parsed = _extract_json_object(content)
        lines = _apply_llm_grouping(normalized, parsed)
        logger.info(
            "Ollama refined %d words into %d karaoke lines (model=%s)",
            len(normalized),
            len(lines),
            model,
        )
        return lines, True
    except Exception as exc:
        logger.warning(
            "Ollama refinement skipped (%s); falling back to deterministic grouping",
            exc,
        )
        return group_words_into_lines(normalized), False
    finally:
        if owns_client:
            http.close()
