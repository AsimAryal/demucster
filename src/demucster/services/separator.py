"""Demucs model wrapper for GPU-accelerated audio stem separation.

Manages model lifecycle, caching, and inference with real-time progress
reporting via the ``demucs.api.Separator`` callback mechanism.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported models — ranked by quality (best first)
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "htdemucs_ft": {
        "description": "Hybrid Transformer Demucs (fine-tuned) — highest quality, 4 stems",
        "stems": ["drums", "bass", "other", "vocals"],
        "is_default": True,
    },
    "htdemucs": {
        "description": "Hybrid Transformer Demucs — fast, strong quality, 4 stems",
        "stems": ["drums", "bass", "other", "vocals"],
    },
    "htdemucs_6s": {
        "description": "Hybrid Transformer Demucs — 6 stems (adds piano + guitar)",
        "stems": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    },
    "mdx_extra": {
        "description": "MDX-Net extra — alternative high-quality profile, 4 stems",
        "stems": ["drums", "bass", "other", "vocals"],
    },
    "mdx_extra_q": {
        "description": "MDX-Net extra (quantised) — lighter, 4 stems",
        "stems": ["drums", "bass", "other", "vocals"],
    },
    "mdx": {
        "description": "MDX-Net baseline — fastest MDX variant, 4 stems",
        "stems": ["drums", "bass", "other", "vocals"],
    },
}


def _demucs_progress_pct(info: dict[str, Any]) -> float:
    """Convert a Demucs 4 callback payload to an approximate percentage."""
    model_count = max(int(info.get("models", 1)), 1)
    model_index = min(max(int(info.get("model_idx_in_bag", 0)), 0), model_count - 1)
    audio_length = max(int(info.get("audio_length", 1)), 1)
    segment_offset = min(max(int(info.get("segment_offset", 0)), 0), audio_length)
    within_model = segment_offset / audio_length
    return min(((model_index + within_model) / model_count) * 100.0, 99.0)


class DemucsService:
    """Manages Demucs model lifecycle and stem separation inference.

    Pre-warms the default model on startup to eliminate cold-start latency.
    Additional models are loaded lazily on first request.  Each call to
    :meth:`separate` creates a fresh ``demucs.api.Separator`` with a
    per-job progress callback so concurrent jobs don't interfere.

    Parameters
    ----------
    device:
        ``torch.device`` to place models and tensors on.
    segment:
        Optional segment override (seconds).  ``None`` lets the model
        choose its default.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        segment: int | None = None,
    ) -> None:
        self._device = device
        self._segment = segment
        self._warmed: set[str] = set()
        logger.info("DemucsService initialised  device=%s  segment=%s", device, segment)

    # -- Warmup --------------------------------------------------------------

    def warmup(self, model_name: str) -> None:
        """Pre-download and cache *model_name* so the first job is fast.

        Raises
        ------
        RuntimeError
            If the model cannot be loaded (e.g. network error on first download).
        """
        if model_name in self._warmed:
            return

        logger.info("Warming up model: %s …", model_name)
        try:
            import demucs.api  # lazy import — heavy dependency

            kwargs: dict[str, Any] = {"model": model_name, "device": str(self._device)}
            if self._segment is not None:
                kwargs["segment"] = self._segment

            sep = demucs.api.Separator(**kwargs)
            del sep
            torch.cuda.empty_cache()
            self._warmed.add(model_name)
            logger.info("Model %s warmed up successfully", model_name)
        except Exception:
            logger.exception("Failed to warm up model %s", model_name)
            raise

    # -- Inference -----------------------------------------------------------

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        model_name: str,
        output_format: str = "mp3",
        mp3_bitrate: int = 320,
        progress_callback: Callable[[float], None] | None = None,
        requested_stems: list[str] | None = None,
        exclude_stems: list[str] | None = None,
    ) -> dict[str, Path]:
        """Run Demucs separation and save stems to disk.

        Parameters
        ----------
        input_path:
            Path to the source audio file.
        output_dir:
            Directory to write separated stem files into.
        model_name:
            Demucs model identifier (must be in ``SUPPORTED_MODELS``).
        output_format:
            ``"mp3"`` or ``"wav"``.
        mp3_bitrate:
            MP3 encoding bitrate in kbps (ignored for WAV).
        progress_callback:
            Optional callable receiving a percentage ``[0, 100]`` as chunks
            are processed.
        requested_stems:
            Subset of stem names to save.  ``None`` saves all stems from the
            model.
        exclude_stems:
            Stems to exclude — for each excluded stem, a mix of all *other*
            stems is saved as ``no_{stem_name}``.  Can be used alongside
            ``requested_stems``.

        Returns
        -------
        dict[str, Path]
            Mapping ``stem_name → output_file_path``.  Stemless mixes use
            the ``no_`` prefix (e.g. ``no_drums``).

        Raises
        ------
        ValueError
            If *model_name* is not supported.
        torch.cuda.OutOfMemoryError
            If the GPU runs out of memory during inference.
        """
        import demucs.api

        from demucster.services.audio import save_stem

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
            )
        if output_format not in {"mp3", "wav"}:
            raise ValueError(f"Unsupported output format: {output_format!r}")
        if not 128 <= mp3_bitrate <= 320:
            raise ValueError("MP3 bitrate must be between 128 and 320 kbps")

        supported_stems = set(SUPPORTED_MODELS[model_name]["stems"])
        if requested_stems is not None:
            invalid_stems = sorted(set(requested_stems) - supported_stems)
            if invalid_stems:
                raise ValueError(f"Stems {invalid_stems} are unavailable for model '{model_name}'")
            if not requested_stems:
                raise ValueError("At least one stem must be requested")

        if exclude_stems is not None:
            invalid_exclude = sorted(set(exclude_stems) - supported_stems)
            if invalid_exclude:
                raise ValueError(
                    f"Exclude stems {invalid_exclude} are unavailable for model '{model_name}'"
                )

        output_dir.mkdir(parents=True, exist_ok=True)

        # -- Demucs progress callback ----------------------------------------
        def _demucs_cb(info: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            progress_callback(_demucs_progress_pct(info))

        logger.info(
            "Starting separation  model=%s  input=%s  format=%s  exclude=%s",
            model_name,
            input_path.name,
            output_format,
            exclude_stems,
        )

        # -- Create per-job Separator ----------------------------------------
        kwargs: dict[str, Any] = {
            "model": model_name,
            "device": str(self._device),
            "callback": _demucs_cb,
            "progress": False,  # we use our own callback, not tqdm
        }
        if self._segment is not None:
            kwargs["segment"] = self._segment

        separator = demucs.api.Separator(**kwargs)
        sample_rate: int = separator.samplerate

        # -- Run inference ---------------------------------------------------
        _origin, separated = separator.separate_audio_file(input_path)

        # -- Save individual stems -------------------------------------------
        available_stems = list(separated.keys())
        stems_to_save = requested_stems if requested_stems is not None else available_stems

        stem_paths: dict[str, Path] = {}
        for stem_name in stems_to_save:
            if stem_name not in separated:
                raise RuntimeError(
                    f"Model '{model_name}' did not produce requested stem '{stem_name}'"
                )

            stem_path = output_dir / f"{stem_name}.{output_format}"
            save_stem(
                tensor=separated[stem_name],
                sample_rate=sample_rate,
                output_path=stem_path,
                output_format=output_format,
                mp3_bitrate=mp3_bitrate,
            )
            stem_paths[stem_name] = stem_path
            logger.debug("Saved stem: %s → %s", stem_name, stem_path)

        # -- Save stemless (minus-one) mixes ---------------------------------
        if exclude_stems:
            for excluded in exclude_stems:
                remaining = [tensor for name, tensor in separated.items() if name != excluded]
                if not remaining:
                    logger.warning("Skipping exclude '%s' — no remaining stems to mix", excluded)
                    continue

                mix_tensor = remaining[0]
                for tensor in remaining[1:]:
                    mix_tensor = mix_tensor + tensor

                mix_name = f"no_{excluded}"
                mix_path = output_dir / f"{mix_name}.{output_format}"
                save_stem(
                    tensor=mix_tensor,
                    sample_rate=sample_rate,
                    output_path=mix_path,
                    output_format=output_format,
                    mp3_bitrate=mp3_bitrate,
                )
                stem_paths[mix_name] = mix_path
                logger.debug("Saved stemless mix: %s → %s", mix_name, mix_path)

        if not stem_paths:
            raise RuntimeError("Demucs produced no output stems")

        # -- Cleanup GPU memory ----------------------------------------------
        del separator, _origin, separated
        torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(100.0)

        logger.info(
            "Separation complete  %d stems → %s",
            len(stem_paths),
            output_dir,
        )
        return stem_paths
