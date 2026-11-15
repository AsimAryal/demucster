"""Whisper / stable-ts lyrics transcription and forced alignment service.

Extracts word-level timestamps from isolated vocal audio on GPU (or CPU),
optionally refines karaoke line breaks via Ollama, and writes ``lyrics.json``,
``lyrics.lrc``, ``lyrics.ass``, and ``lyrics.srt`` artifacts.
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING, Any

import torch

from demucster.models.schemas import LyricsSummary, LyricsWord
from demucster.services.lyrics_formats import build_lyrics_document, write_lyrics_artifacts
from demucster.services.ollama_refiner import refine_lines_with_ollama

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_LYRICS_MODELS: dict[str, dict[str, Any]] = {
    "large-v3-turbo": {
        "description": "Whisper large-v3-turbo — fast, high-accuracy multilingual transcription",
        "is_default": True,
    },
    "large-v3": {
        "description": "Whisper large-v3 — maximum accuracy for complex vocals and languages",
    },
    "medium": {
        "description": "Whisper medium — lighter footprint for faster turnaround",
    },
}


def _ensure_torchaudio_compat() -> None:
    """Register a lightweight ``torchaudio`` shim when absent (e.g. NGC containers).

    NVIDIA NGC PyTorch images omit ``torchaudio``, yet ``stable_whisper`` imports
    it at module load time for ``functional.resample``, biquad filters, and
    ``save``/``load``. Implementing those primitives with ``julius`` and
    ``soundfile`` preserves NGC's Blackwell PyTorch stack untouched.
    """
    existing = sys.modules.get("torchaudio")
    if existing is not None:
        return
    try:
        import torchaudio  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import julius
    import soundfile as sf

    def _resample(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if int(orig_freq) == int(new_freq):
            return waveform
        return julius.resample_frac(waveform, int(orig_freq), int(new_freq))

    def _clamp_cutoff(cutoff_freq: float, sample_rate: int) -> float:
        return max(1e-4, min(0.499, float(cutoff_freq) / float(sample_rate)))

    def _highpass_biquad(
        waveform: torch.Tensor,
        sample_rate: int,
        cutoff_freq: float,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return julius.highpass_filter(waveform, _clamp_cutoff(cutoff_freq, sample_rate))

    def _lowpass_biquad(
        waveform: torch.Tensor,
        sample_rate: int,
        cutoff_freq: float,
        **_kwargs: Any,
    ) -> torch.Tensor:
        return julius.lowpass_filter(waveform, _clamp_cutoff(cutoff_freq, sample_rate))

    def _save(
        uri: Any,
        src: torch.Tensor,
        sample_rate: int,
        **_kwargs: Any,
    ) -> None:
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, int(sample_rate))

    def _load(uri: Any, **_kwargs: Any) -> tuple[torch.Tensor, int]:
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), int(sr)

    functional_mod = types.ModuleType("torchaudio.functional")
    functional_mod.__spec__ = ModuleSpec("torchaudio.functional", None)
    functional_mod.resample = _resample  # type: ignore[attr-defined]
    functional_mod.highpass_biquad = _highpass_biquad  # type: ignore[attr-defined]
    functional_mod.lowpass_biquad = _lowpass_biquad  # type: ignore[attr-defined]

    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.__spec__ = ModuleSpec("torchaudio", None)
    torchaudio_mod.__version__ = "2.0.0+demucster-shim"  # type: ignore[attr-defined]
    torchaudio_mod.functional = functional_mod  # type: ignore[attr-defined]
    torchaudio_mod.save = _save  # type: ignore[attr-defined]
    torchaudio_mod.load = _load  # type: ignore[attr-defined]
    torchaudio_mod.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]

    sys.modules["torchaudio.functional"] = functional_mod
    sys.modules["torchaudio"] = torchaudio_mod
    logger.info("Registered julius/soundfile compatibility shim for missing torchaudio")


def _words_from_result(result: Any) -> tuple[list[LyricsWord], str | None, float]:
    """Extract flattened ``LyricsWord`` tokens, detected language, and duration."""
    words: list[LyricsWord] = []
    segments = getattr(result, "segments", None) or []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        if seg_words:
            for item in seg_words:
                text = getattr(item, "word", "")
                start = getattr(item, "start", None)
                end = getattr(item, "end", None)
                prob = getattr(item, "probability", None)
                if isinstance(text, str) and start is not None and end is not None:
                    score = max(0.0, min(1.0, float(prob))) if prob is not None else None
                    words.append(
                        LyricsWord(
                            word=text,
                            start=float(start),
                            end=float(end),
                            score=score,
                        )
                    )
        else:
            seg_text = getattr(seg, "text", "")
            seg_start = getattr(seg, "start", None)
            seg_end = getattr(seg, "end", None)
            if isinstance(seg_text, str) and seg_start is not None and seg_end is not None:
                tokens = seg_text.split()
                if tokens:
                    span = max(0.05, float(seg_end) - float(seg_start))
                    step = span / len(tokens)
                    for idx, token in enumerate(tokens):
                        w_start = float(seg_start) + idx * step
                        w_end = w_start + step
                        words.append(LyricsWord(word=token, start=w_start, end=w_end))

    language = getattr(result, "language", None)
    if not isinstance(language, str) or not language.strip():
        language = None
    duration = float(getattr(result, "duration", 0.0) or 0.0)
    if duration <= 0.0 and words:
        duration = words[-1].end
    return words, language, duration


class LyricsService:
    """Manages Whisper/stable-ts model lifecycle and lyrics timestamp extraction."""

    def __init__(
        self,
        device: torch.device,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self._device = device
        self._cache_dir = cache_dir
        self._models: dict[str, Any] = {}
        self._model_lock = threading.Lock()
        logger.info("LyricsService initialised  device=%s  cache_dir=%s", device, cache_dir)

    def _get_model(self, model_name: str) -> Any:
        """Load and cache a ``stable_whisper`` model on the configured device."""
        with self._model_lock:
            cached = self._models.get(model_name)
            if cached is not None:
                return cached

            _ensure_torchaudio_compat()
            import stable_whisper  # lazy import — heavy dependency

            kwargs: dict[str, Any] = {"device": str(self._device)}
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                kwargs["download_root"] = str(self._cache_dir)

            logger.info("Loading Whisper model %s on %s …", model_name, self._device)
            model = stable_whisper.load_model(model_name, **kwargs)
            self._models[model_name] = model
            return model

    def warmup(self, model_name: str) -> None:
        """Pre-load *model_name* so the first lyrics job avoids cold-start latency."""
        if model_name not in SUPPORTED_LYRICS_MODELS:
            raise ValueError(f"Unsupported lyrics model '{model_name}'")
        self._get_model(model_name)
        logger.info("Lyrics model %s warmed up successfully", model_name)

    def extract(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        model_name: str,
        language: str | None = None,
        reference_lyrics: str | None = None,
        title: str | None = None,
        use_llm_cleanup: bool = True,
        ollama_url: str | None = None,
        ollama_model: str = "qwen3:27b",
        ollama_timeout_seconds: float = 120.0,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[dict[str, Path], LyricsSummary]:
        """Extract word-level timestamped lyrics and write karaoke artifact files."""
        if model_name not in SUPPORTED_LYRICS_MODELS:
            raise ValueError(
                f"Unsupported lyrics model '{model_name}'. "
                f"Choose from: {', '.join(SUPPORTED_LYRICS_MODELS)}"
            )

        if progress_callback:
            progress_callback(10.0)

        model = self._get_model(model_name)

        if progress_callback:
            progress_callback(25.0)

        aligned_with_reference = bool(reference_lyrics and reference_lyrics.strip())
        if aligned_with_reference:
            assert reference_lyrics is not None
            result = model.align(
                str(input_path),
                reference_lyrics,
                language=language or "en",
            )
        else:
            transcribe_kwargs: dict[str, Any] = {
                "vad": True,
                "regroup": True,
                "word_timestamps": True,
            }
            if language:
                transcribe_kwargs["language"] = language
            result = model.transcribe(str(input_path), **transcribe_kwargs)

        if progress_callback:
            progress_callback(65.0)

        words, detected_language, duration = _words_from_result(result)
        effective_language = language or detected_language

        lines, llm_refined = refine_lines_with_ollama(
            words,
            ollama_url=ollama_url if use_llm_cleanup else None,
            model=ollama_model,
            title=title,
            reference_lyrics=reference_lyrics,
            timeout_seconds=ollama_timeout_seconds,
        )

        if progress_callback:
            progress_callback(85.0)

        document = build_lyrics_document(
            lines=lines,
            model=model_name,
            title=title,
            language=effective_language,
            duration_seconds=duration,
            llm_refined=llm_refined,
            llm_model=ollama_model if llm_refined else None,
            aligned_with_reference=aligned_with_reference,
        )
        artifacts = write_lyrics_artifacts(document, output_dir)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(100.0)

        logger.info(
            "Lyrics extraction complete  lines=%d  words=%d  llm_refined=%s → %s",
            document.summary.line_count,
            document.summary.word_count,
            llm_refined,
            output_dir,
        )
        return artifacts, document.summary
