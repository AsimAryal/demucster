"""Tests for optional Ollama Qwen3 karaoke line refinement."""

from __future__ import annotations

import json

import httpx

from demucster.models.schemas import LyricsWord
from demucster.services.ollama_refiner import refine_lines_with_ollama


def _sample_words() -> list[LyricsWord]:
    return [
        LyricsWord(word="Helo", start=0.5, end=0.9, score=0.8),
        LyricsWord(word="world", start=0.95, end=1.3, score=0.9),
        LyricsWord(word="tonight", start=2.2, end=2.7, score=0.95),
    ]


def test_ollama_refiner_skips_when_url_unset() -> None:
    lines, refined = refine_lines_with_ollama(_sample_words(), ollama_url=None)
    assert refined is False
    assert len(lines) == 2
    assert lines[0].text == "Helo world"


def test_ollama_refiner_applies_valid_grouping_and_corrections() -> None:
    payload = {
        "message": {
            "content": json.dumps(
                {
                    "lines": [
                        {
                            "word_indices": [0, 1],
                            "corrected_words": ["Hello", "world,"],
                        },
                        {
                            "word_indices": [2],
                            "corrected_words": ["tonight."],
                        },
                    ]
                }
            )
        }
    }
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=payload))
    with httpx.Client(base_url="http://ollama.local", transport=transport) as http:
        lines, refined = refine_lines_with_ollama(
            _sample_words(),
            ollama_url="http://ollama.local",
            model="qwen3:27b",
            title="Test",
            client=http,
        )

    assert refined is True
    assert len(lines) == 2
    assert lines[0].text == "Hello world,"
    assert lines[0].words[0].start == 0.5
    assert lines[0].words[0].end == 0.9
    assert lines[1].text == "tonight."


def test_ollama_refiner_falls_back_on_bad_indices_or_error() -> None:
    bad_payload = {
        "message": {
            "content": json.dumps({"lines": [{"word_indices": [0, 2]}]}),
        }
    }
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, json=bad_payload))
    with httpx.Client(base_url="http://ollama.local", transport=transport) as http:
        lines, refined = refine_lines_with_ollama(
            _sample_words(),
            ollama_url="http://ollama.local",
            client=http,
        )
    assert refined is False
    assert sum(len(line.words) for line in lines) == 3

    err_transport = httpx.MockTransport(lambda _req: httpx.Response(500))
    with httpx.Client(base_url="http://ollama.local", transport=err_transport) as http:
        lines, refined = refine_lines_with_ollama(
            _sample_words(),
            ollama_url="http://ollama.local",
            client=http,
        )
    assert refined is False
    assert len(lines) == 2
