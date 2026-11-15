"""Karaoke lyrics line segmentation and artifact formatting utilities.

Generates four synchronized output formats from word-level timestamps:
* ``lyrics.json`` — Canonical structured document (lines + words + summary)
* ``lyrics.lrc``  — Enhanced LRC with inline ``<mm:ss.xx>`` word timestamps
* ``lyrics.ass``  — Advanced SubStation Alpha with ``\\kf`` sweep tags
* ``lyrics.srt``  — Standard line-level SubRip subtitles
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from demucster.models.schemas import LyricsDocument, LyricsLine, LyricsSummary, LyricsWord

if TYPE_CHECKING:
    from pathlib import Path

_STRONG_END_RE = re.compile(r"[.!?]+[\"'\u201d\u2019)]*$")


def _format_lrc_timestamp(seconds: float) -> str:
    """Format *seconds* as ``MM:SS.xx`` with proper centisecond carry."""
    total_cs = max(0, round(seconds * 100))
    minutes, rem_cs = divmod(total_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"{minutes:02d}:{secs:02d}.{cs:02d}"


def _format_ass_timestamp(seconds: float) -> str:
    """Format *seconds* as ``H:MM:SS.cc`` for ASS dialogue events."""
    total_cs = max(0, round(seconds * 100))
    hours, rem_cs = divmod(total_cs, 360_000)
    minutes, rem_cs = divmod(rem_cs, 6000)
    secs, cs = divmod(rem_cs, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _format_srt_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS,mmm`` for SubRip subtitles."""
    total_ms = max(0, round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters that carry special meaning in ASS override blocks."""
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def normalize_words(words: list[LyricsWord]) -> list[LyricsWord]:
    """Sanitize raw ASR/aligned words into clean, monotonic word tokens."""
    cleaned: list[LyricsWord] = []
    prev_end = 0.0
    for item in words:
        token = " ".join(item.word.split()).strip()
        if not token:
            continue
        start = max(0.0, round(float(item.start), 3))
        end = max(start + 0.02, round(float(item.end), 3))
        if start < prev_end:
            start = prev_end
            if end <= start:
                end = round(start + 0.02, 3)
        prev_end = end
        cleaned.append(
            LyricsWord(
                word=token,
                start=start,
                end=end,
                score=round(float(item.score), 4) if item.score is not None else None,
            )
        )
    return cleaned


def group_words_into_lines(
    words: list[LyricsWord],
    *,
    max_chars: int = 42,
    pause_threshold_s: float = 0.65,
    max_words: int = 9,
) -> list[LyricsLine]:
    """Deterministically group timed words into singable karaoke lines."""
    normalized = normalize_words(words)
    if not normalized:
        return []

    lines: list[LyricsLine] = []
    bucket: list[LyricsWord] = []
    char_len = 0

    def _flush() -> None:
        nonlocal bucket, char_len
        if not bucket:
            return
        lines.append(
            LyricsLine(
                index=len(lines),
                start=bucket[0].start,
                end=bucket[-1].end,
                text=" ".join(w.word for w in bucket),
                words=list(bucket),
            )
        )
        bucket = []
        char_len = 0

    for word in normalized:
        if bucket:
            prev = bucket[-1]
            gap = word.start - prev.end
            projected_len = char_len + 1 + len(word.word)
            ends_sentence = bool(_STRONG_END_RE.search(prev.word)) and len(bucket) >= 2
            if (
                gap >= pause_threshold_s
                or ends_sentence
                or projected_len > max_chars
                or len(bucket) >= max_words
            ):
                _flush()

        bucket.append(word)
        char_len = len(word.word) if len(bucket) == 1 else char_len + 1 + len(word.word)

    _flush()
    return lines


def build_lyrics_document(
    *,
    lines: list[LyricsLine],
    model: str,
    title: str | None = None,
    language: str | None = None,
    duration_seconds: float | None = None,
    llm_refined: bool = False,
    llm_model: str | None = None,
    aligned_with_reference: bool = False,
) -> LyricsDocument:
    """Assemble a validated ``LyricsDocument`` and compute summary metrics."""
    word_count = sum(len(line.words) for line in lines)
    inferred_duration = lines[-1].end if lines else 0.0
    final_duration = round(max(duration_seconds or 0.0, inferred_duration), 3)

    summary = LyricsSummary(
        language=language,
        line_count=len(lines),
        word_count=word_count,
        duration_seconds=final_duration,
        llm_refined=llm_refined,
        llm_model=llm_model if llm_refined else None,
        aligned_with_reference=aligned_with_reference,
    )
    return LyricsDocument(
        version=1,
        title=title,
        model=model,
        summary=summary,
        lines=lines,
    )


def format_lrc(document: LyricsDocument) -> str:
    """Render *document* as Enhanced LRC with inline word timestamps."""
    rows: list[str] = []
    if document.title:
        rows.append(f"[ti:{document.title}]")
    if document.summary.language:
        rows.append(f"[la:{document.summary.language}]")
    rows.append("[re:Demucster]")

    for line in document.lines:
        line_tag = f"[{_format_lrc_timestamp(line.start)}]"
        if line.words:
            parts = [f"<{_format_lrc_timestamp(w.start)}>{w.word}" for w in line.words]
            rows.append(f"{line_tag} {' '.join(parts)}")
        else:
            rows.append(f"{line_tag} {line.text}")

    return "\n".join(rows) + ("\n" if rows else "")


def format_ass(document: LyricsDocument) -> str:
    """Render *document* as ASS subtitles with ``\\kf`` progressive karaoke sweep."""
    title = (document.title or "Demucster Karaoke").replace("\n", " ").strip()
    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Karaoke,Arial,64,&H005BE4D6,&H00FFFFFF,&H00121826,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,3,2,2,80,80,90,1"
        ),
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"),
    ]

    events: list[str] = []
    for line in document.lines:
        start_str = _format_ass_timestamp(line.start)
        end_str = _format_ass_timestamp(max(line.end, line.start + 0.1))
        if not line.words:
            text_payload = _escape_ass_text(line.text)
        else:
            cursor_cs = round(line.start * 100)
            chunks: list[str] = []
            for idx, word in enumerate(line.words):
                word_start_cs = round(word.start * 100)
                word_end_cs = max(word_start_cs + 2, round(word.end * 100))
                gap_cs = word_start_cs - cursor_cs
                if gap_cs > 0:
                    chunks.append(f"{{\\k{gap_cs}}}")
                dur_cs = word_end_cs - word_start_cs
                suffix = " " if idx < len(line.words) - 1 else ""
                chunks.append(f"{{\\kf{dur_cs}}}{_escape_ass_text(word.word)}{suffix}")
                cursor_cs = word_end_cs
            text_payload = "".join(chunks)

        events.append(f"Dialogue: 0,{start_str},{end_str},Karaoke,,0,0,0,karaoke,{text_payload}")

    return "\n".join([*header, *events]) + "\n"


def format_srt(document: LyricsDocument) -> str:
    """Render *document* as line-level SubRip (``.srt``) subtitles."""
    blocks: list[str] = []
    for idx, line in enumerate(document.lines, start=1):
        start_str = _format_srt_timestamp(line.start)
        end_str = _format_srt_timestamp(max(line.end, line.start + 0.1))
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{line.text}\n")
    return "\n".join(blocks)


def write_lyrics_artifacts(document: LyricsDocument, output_dir: Path) -> dict[str, Path]:
    """Write ``lyrics.json``, ``lyrics.lrc``, ``lyrics.ass``, and ``lyrics.srt``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[str, str]] = {
        "json": ("lyrics.json", document.model_dump_json(indent=2)),
        "lrc": ("lyrics.lrc", format_lrc(document)),
        "ass": ("lyrics.ass", format_ass(document)),
        "srt": ("lyrics.srt", format_srt(document)),
    }
    paths: dict[str, Path] = {}
    for kind, (filename, content) in files.items():
        target = output_dir / filename
        target.write_text(content, encoding="utf-8")
        paths[kind] = target
    return paths
