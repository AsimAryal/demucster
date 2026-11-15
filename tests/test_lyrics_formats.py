"""Unit tests for karaoke lyrics grouping and formatters (JSON, LRC, ASS, SRT)."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import torch

from demucster.models.schemas import LyricsWord
from demucster.services.lyrics import _ensure_torchaudio_compat
from demucster.services.lyrics_formats import (
    _format_ass_timestamp,
    _format_lrc_timestamp,
    build_lyrics_document,
    format_ass,
    format_lrc,
    format_srt,
    group_words_into_lines,
    normalize_words,
    write_lyrics_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_lrc_timestamp_carry_rounding() -> None:
    assert _format_lrc_timestamp(59.996) == "01:00.00"
    assert _format_lrc_timestamp(0.0) == "00:00.00"
    assert _format_ass_timestamp(65.5) == "0:01:05.50"


def test_normalize_words_enforces_monotonic_timestamps() -> None:
    words = [
        LyricsWord(word="  Hello  ", start=1.0, end=1.5),
        LyricsWord(word="   ", start=1.6, end=1.8),
        LyricsWord(word="world", start=1.4, end=1.45),
    ]
    cleaned = normalize_words(words)
    assert [w.word for w in cleaned] == ["Hello", "world"]
    assert cleaned[1].start >= cleaned[0].end
    assert cleaned[1].end > cleaned[1].start


def test_group_words_breaks_on_pauses_and_length() -> None:
    words = [
        LyricsWord(word="First", start=0.5, end=0.8),
        LyricsWord(word="line.", start=0.85, end=1.2),
        LyricsWord(word="Second", start=2.5, end=2.9),
        LyricsWord(word="line", start=2.95, end=3.3),
    ]
    lines = group_words_into_lines(words, pause_threshold_s=0.65)
    assert len(lines) == 2
    assert lines[0].text == "First line."
    assert lines[1].text == "Second line"


def test_formatters_and_artifact_writer(tmp_path: Path) -> None:
    words = [
        LyricsWord(word="Sing", start=1.0, end=1.4, score=0.95),
        LyricsWord(word="{loud}", start=1.6, end=2.1, score=0.91),
    ]
    lines = group_words_into_lines(words)
    doc = build_lyrics_document(
        lines=lines,
        model="large-v3-turbo",
        title="Test Song",
        language="en",
        duration_seconds=2.5,
        llm_refined=True,
        llm_model="qwen3:27b",
    )

    lrc = format_lrc(doc)
    assert "[ti:Test Song]" in lrc
    assert "[00:01.00] <00:01.00>Sing <00:01.60>{loud}" in lrc

    ass = format_ass(doc)
    assert "[V4+ Styles]" in ass
    assert "{\\kf40}Sing {\\k20}{\\kf50}\\{loud\\}" in ass

    srt = format_srt(doc)
    assert "00:00:01,000 --> 00:00:02,100" in srt
    assert "Sing {loud}" in srt

    paths = write_lyrics_artifacts(doc, tmp_path / "out")
    assert set(paths) == {"json", "lrc", "ass", "srt"}
    parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert parsed["summary"]["word_count"] == 2
    assert parsed["summary"]["llm_model"] == "qwen3:27b"


def test_torchaudio_compat_shim_allows_stable_whisper_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    monkeypatch.delitem(sys.modules, "torchaudio.functional", raising=False)
    for name in list(sys.modules):
        if name == "stable_whisper" or name.startswith("stable_whisper."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    _ensure_torchaudio_compat()

    import stable_whisper
    import torchaudio

    assert hasattr(stable_whisper, "load_model")
    wave = torch.randn(1, 1600)
    resampled = torchaudio.functional.resample(wave, 16000, 8000)
    assert resampled.shape == (1, 800)

    filtered = torchaudio.functional.highpass_biquad(
        torchaudio.functional.lowpass_biquad(wave, 16000, 5000),
        16000,
        200,
    )
    assert filtered.shape == wave.shape

    wav_path = tmp_path / "shim.wav"
    torchaudio.save(wav_path, wave, 16000)
    loaded, sr = torchaudio.load(wav_path)
    assert sr == 16000
    assert loaded.shape == wave.shape
