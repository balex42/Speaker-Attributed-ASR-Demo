"""Core pipeline combining NVIDIA Nemotron-3-Diarization and Parakeet-TDT-0.6B-v3.

- Diarization: `nvidia/Nemotron-3-Diarization` (`SortformerEncLabelModel`)
- ASR + Word Timestamps: `nvidia/parakeet-tdt-0.6b-v3` (`ASRModel`)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DIARIZATION_MODEL_ID = "nvidia/Nemotron-3-Diarization"
ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
TARGET_SAMPLE_RATE = 16000

# Sortformer latency / context presets (1 frame = 80ms after subsampling)
DIARIZATION_PRESETS: dict[str, dict[str, Any]] = {
    "offline_high_accuracy": {
        "label": "Offline / High Accuracy (30.4s window)",
        "chunk_len": 340,
        "chunk_right_context": 40,
        "fifo_len": 40,
    },
    "streaming_balanced": {
        "label": "Streaming Balanced (~2.5s latency)",
        "chunk_len": 16,
        "chunk_right_context": 16,
        "fifo_len": 128,
    },
    "streaming_low_latency": {
        "label": "Streaming Low Latency (~0.32s latency)",
        "chunk_len": 4,
        "chunk_right_context": 0,
        "fifo_len": 188,
    },
}


@dataclass
class DiarizationSegment:
    """A single speaker activity interval from Nemotron-3-Diarization."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class WordTiming:
    """A single transcribed word with start/end timestamps and assigned speaker."""

    word: str
    start: float
    end: float
    speaker: str = "speaker_0"
    overlapping_speakers: list[str] = field(default_factory=list)


@dataclass
class SpeakerTurn:
    """A contiguous conversational turn spoken by a single speaker."""

    speaker: str
    start: float
    end: float
    text: str
    words: list[WordTiming] = field(default_factory=list)
    has_overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerStats:
    """Aggregated speaking statistics for a single speaker."""

    speaker: str
    total_duration: float
    percentage: float
    turn_count: int
    word_count: int


@dataclass
class PipelineResult:
    """Complete result from running Diarization + ASR alignment."""

    audio_duration: float
    processing_time: float
    full_text: str
    diarization_segments: list[DiarizationSegment]
    words: list[WordTiming]
    turns: list[SpeakerTurn]
    speaker_stats: list[SpeakerStats]
    diarization_model: str = DIARIZATION_MODEL_ID
    asr_model: str = ASR_MODEL_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def format_timestamp(seconds: float, srt: bool = False) -> str:
    """Format seconds into MM:SS.mm or HH:MM:SS,mmm (for SRT)."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if srt:
        whole_secs = int(secs)
        millis = int(round((secs - whole_secs) * 1000))
        if millis >= 1000:
            whole_secs += 1
            millis -= 1000
        return f"{hours:02d}:{minutes:02d}:{whole_secs:02d},{millis:03d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def normalize_speaker_label(raw_label: str) -> str:
    """Normalize speaker identifiers such as '0', 'speaker_0', 'SPEAKER_00' into 'Speaker 1'."""
    cleaned = raw_label.strip()
    match = re.search(r"(\d+)$", cleaned)
    if match:
        idx = int(match.group(1))
        return f"Speaker {idx + 1}"
    return cleaned or "Speaker 1"


def parse_diarization_segments(raw_output: Any) -> list[DiarizationSegment]:
    """Parse output from `SortformerEncLabelModel.diarize()` into `DiarizationSegment` objects.

    Supports nested lists of strings (`"0.120, 1.450, speaker_0"` or `"0.120 1.450 speaker_0"`),
    RTTM lines (`"SPEAKER file 1 0.120 1.330 <NA> <NA> speaker_0 <NA> <NA>"`),
    dictionaries, or tuples.
    """
    if raw_output is None:
        return []

    # Unwrap outer batch list if `raw_output` is `[[seg1, seg2, ...]]`
    items = raw_output
    if (
        isinstance(items, (list, tuple))
        and len(items) == 1
        and isinstance(items[0], (list, tuple))
        and (len(items[0]) == 0 or not isinstance(items[0][0], (int, float)))
    ):
        items = items[0]

    segments: list[DiarizationSegment] = []
    csv_or_space_pattern = re.compile(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:,|\s)\s*([+-]?\d+(?:\.\d+)?)\s*(?:,|\s)\s*(.+?)\s*$"
    )

    for entry in items:
        if isinstance(entry, DiarizationSegment):
            segments.append(entry)
        elif isinstance(entry, dict):
            start = float(entry.get("start", entry.get("begin", 0.0)))
            end = float(entry.get("end", start + float(entry.get("duration", 0.0))))
            speaker = normalize_speaker_label(str(entry.get("speaker", entry.get("label", "speaker_0"))))
            if end > start:
                segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
            start = float(entry[0])
            end = float(entry[1])
            speaker = normalize_speaker_label(str(entry[2]))
            if end > start:
                segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))
        elif isinstance(entry, str):
            line = entry.strip()
            if not line:
                continue
            if line.upper().startswith("SPEAKER "):
                parts = line.split()
                if len(parts) >= 8:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = normalize_speaker_label(parts[7])
                    if duration > 0:
                        segments.append(
                            DiarizationSegment(start=start, end=start + duration, speaker=speaker)
                        )
                    continue
            match = csv_or_space_pattern.match(line)
            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                speaker = normalize_speaker_label(match.group(3))
                if end > start:
                    segments.append(DiarizationSegment(start=start, end=end, speaker=speaker))

    segments.sort(key=lambda s: (s.start, s.end, s.speaker))
    return segments


def extract_word_timings(asr_hypothesis: Any) -> tuple[str, list[WordTiming]]:
    """Extract full text and word-level (or segment-level fallback) timestamps from Parakeet output."""
    if isinstance(asr_hypothesis, list) and len(asr_hypothesis) > 0:
        asr_hypothesis = asr_hypothesis[0]

    if isinstance(asr_hypothesis, str):
        return asr_hypothesis, []

    full_text = getattr(asr_hypothesis, "text", "") or ""
    timestamp_dict = getattr(asr_hypothesis, "timestamp", None) or getattr(
        asr_hypothesis, "timestep", None
    )

    words: list[WordTiming] = []
    if isinstance(timestamp_dict, dict):
        raw_words = timestamp_dict.get("word") or []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            word_str = str(item.get("word", item.get("char", ""))).strip()
            if not word_str:
                continue
            start = float(item.get("start", item.get("start_offset", 0.0)))
            end = float(item.get("end", item.get("end_offset", start)))
            words.append(WordTiming(word=word_str, start=start, end=max(start, end)))

        # Fallback to segment timestamps if word timestamps are empty
        if not words and timestamp_dict.get("segment"):
            for seg in timestamp_dict["segment"]:
                if not isinstance(seg, dict):
                    continue
                seg_text = str(seg.get("segment", seg.get("text", ""))).strip()
                if not seg_text:
                    continue
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                words.append(WordTiming(word=seg_text, start=start, end=max(start, end)))

    return full_text, words


def align_words_with_speakers(
    words: list[WordTiming],
    diar_segments: list[DiarizationSegment],
    nearest_tolerance: float = 1.5,
    merge_gap_threshold: float = 1.2,
) -> tuple[list[WordTiming], list[SpeakerTurn]]:
    """Align word-level ASR timestamps with speaker diarization intervals.

    1. For each word `[w_start, w_end]`, computes overlap duration with every active
       `DiarizationSegment`.
    2. Assigns the speaker with the highest temporal overlap.
    3. If a word has zero direct overlap with any segment, assigns the temporally closest
       diarization segment within `nearest_tolerance` seconds (or the nearest overall).
    4. Groups consecutive words assigned to the same speaker into `SpeakerTurn` objects.
    """
    if not words:
        # If ASR returned no word timestamps, build turns directly from diarization segments
        turns = [
            SpeakerTurn(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                text="",
                words=[],
            )
            for seg in diar_segments
        ]
        return [], turns

    if not diar_segments:
        aligned_words = [
            WordTiming(word=w.word, start=w.start, end=w.end, speaker="Speaker 1")
            for w in words
        ]
        turn = SpeakerTurn(
            speaker="Speaker 1",
            start=aligned_words[0].start,
            end=aligned_words[-1].end,
            text=" ".join(w.word for w in aligned_words),
            words=aligned_words,
        )
        return aligned_words, [turn]

    aligned_words: list[WordTiming] = []
    eps = 1e-3

    for w in words:
        w_start = w.start
        w_end = max(w.end, w_start + eps)
        w_mid = 0.5 * (w_start + w_end)

        speaker_overlaps: dict[str, float] = {}
        for seg in diar_segments:
            overlap_start = max(w_start, seg.start)
            overlap_end = min(w_end, seg.end)
            overlap = overlap_end - overlap_start
            if overlap > 0:
                speaker_overlaps[seg.speaker] = speaker_overlaps.get(seg.speaker, 0.0) + overlap

        if speaker_overlaps:
            ranked = sorted(speaker_overlaps.items(), key=lambda kv: (-kv[1], kv[0]))
            best_speaker = ranked[0][0]
            overlapping = [spk for spk, dur in ranked if dur > 0.02]
        else:
            # Find nearest segment by distance to word boundaries / midpoint
            best_speaker = diar_segments[0].speaker
            best_dist = float("inf")
            for seg in diar_segments:
                if w_end < seg.start:
                    dist = seg.start - w_end
                elif w_start > seg.end:
                    dist = w_start - seg.end
                else:
                    dist = abs(w_mid - 0.5 * (seg.start + seg.end))
                if dist < best_dist:
                    best_dist = dist
                    best_speaker = seg.speaker
            if best_dist > nearest_tolerance and aligned_words:
                # Prefer continuing previous speaker if far from all segments
                best_speaker = aligned_words[-1].speaker
            overlapping = []

        aligned_words.append(
            WordTiming(
                word=w.word,
                start=w.start,
                end=w.end,
                speaker=best_speaker,
                overlapping_speakers=overlapping if len(overlapping) > 1 else [],
            )
        )

    # Group consecutive words with the same speaker into SpeakerTurns
    turns: list[SpeakerTurn] = []
    for aw in aligned_words:
        if (
            turns
            and turns[-1].speaker == aw.speaker
            and (aw.start - turns[-1].end) <= merge_gap_threshold
        ):
            current = turns[-1]
            current.end = max(current.end, aw.end)
            current.words.append(aw)
            current.text = f"{current.text} {aw.word}".strip()
            if aw.overlapping_speakers:
                current.has_overlap = True
        else:
            turns.append(
                SpeakerTurn(
                    speaker=aw.speaker,
                    start=aw.start,
                    end=aw.end,
                    text=aw.word,
                    words=[aw],
                    has_overlap=bool(aw.overlapping_speakers),
                )
            )

    return aligned_words, turns


def compute_speaker_stats(
    diar_segments: list[DiarizationSegment],
    turns: list[SpeakerTurn],
) -> list[SpeakerStats]:
    """Compute per-speaker speaking duration, share percentage, turn count, and word count."""
    durations: dict[str, float] = {}
    for seg in diar_segments:
        durations[seg.speaker] = durations.get(seg.speaker, 0.0) + seg.duration

    turn_counts: dict[str, int] = {}
    word_counts: dict[str, int] = {}
    for turn in turns:
        turn_counts[turn.speaker] = turn_counts.get(turn.speaker, 0) + 1
        word_counts[turn.speaker] = word_counts.get(turn.speaker, 0) + len(turn.words)
        if turn.speaker not in durations:
            durations[turn.speaker] = durations.get(turn.speaker, 0.0) + turn.duration

    total_speech = sum(durations.values())
    all_speakers = sorted(set(durations.keys()) | set(turn_counts.keys()))

    stats: list[SpeakerStats] = []
    for spk in all_speakers:
        dur = durations.get(spk, 0.0)
        pct = (dur / total_speech * 100.0) if total_speech > 0 else 0.0
        stats.append(
            SpeakerStats(
                speaker=spk,
                total_duration=round(dur, 2),
                percentage=round(pct, 1),
                turn_count=turn_counts.get(spk, 0),
                word_count=word_counts.get(spk, 0),
            )
        )
    return stats


def format_turns_as_markdown(turns: list[SpeakerTurn]) -> str:
    """Format speaker turns as a clean Markdown transcript."""
    if not turns:
        return "_No speech detected._"
    lines: list[str] = []
    for turn in turns:
        ts = f"[{format_timestamp(turn.start)} - {format_timestamp(turn.end)}]"
        overlap_tag = " *(overlapping speech)*" if turn.has_overlap else ""
        lines.append(f"**{turn.speaker}** `{ts}`{overlap_tag}: {turn.text}")
    return "\n\n".join(lines)


def format_turns_as_srt(turns: list[SpeakerTurn]) -> str:
    """Format speaker turns as standard SubRip (.srt) subtitles."""
    blocks: list[str] = []
    for idx, turn in enumerate(turns, start=1):
        start_srt = format_timestamp(turn.start, srt=True)
        end_srt = format_timestamp(max(turn.end, turn.start + 0.2), srt=True)
        blocks.append(f"{idx}\n{start_srt} --> {end_srt}\n[{turn.speaker}] {turn.text}\n")
    return "\n".join(blocks)


def format_segments_as_rttm(
    segments: list[DiarizationSegment], file_id: str = "audio"
) -> str:
    """Format diarization segments in NIST RTTM format."""
    lines: list[str] = []
    clean_id = re.sub(r"\s+", "_", Path(file_id).stem or "audio")
    for seg in segments:
        spk_id = seg.speaker.lower().replace(" ", "_")
        lines.append(
            f"SPEAKER {clean_id} 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {spk_id} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def preprocess_audio(input_path: str | Path) -> tuple[str, float]:
    """Convert any audio file to 16 kHz mono WAV required by Nemotron-3-Diarization & Parakeet.

    Returns `(wav_path, duration_seconds)`.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    audio_data, _ = librosa.load(str(input_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)

    duration = float(len(audio_data)) / float(TARGET_SAMPLE_RATE)
    tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix="_16k_mono.wav", prefix="nemo_demo_")
    os.close(tmp_fd)
    sf.write(tmp_wav_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return tmp_wav_path, duration


class DiarizationASRPipeline:
    """End-to-end pipeline loading `nvidia/Nemotron-3-Diarization` and `nvidia/parakeet-tdt-0.6b-v3`."""

    def __init__(
        self,
        diarization_model_name: str = DIARIZATION_MODEL_ID,
        asr_model_name: str = ASR_MODEL_ID,
        device: str | None = None,
    ) -> None:
        self.diarization_model_name = diarization_model_name
        self.asr_model_name = asr_model_name
        self.device = device
        self._diar_model: Any = None
        self._asr_model: Any = None

    def _resolve_device(self) -> str:
        import torch

        if self.device:
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_models(self) -> None:
        """Lazily load both NeMo models onto the target device."""
        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import SortformerEncLabelModel

        target_device = self._resolve_device()
        map_loc = torch.device(target_device)

        if self._diar_model is None:
            logger.info("Loading diarization model: %s on %s", self.diarization_model_name, target_device)
            self._diar_model = SortformerEncLabelModel.from_pretrained(
                self.diarization_model_name,
                map_location=map_loc,
            )
            self._diar_model.eval()
            self._diar_model = self._diar_model.to(map_loc)

        if self._asr_model is None:
            logger.info("Loading ASR model: %s on %s", self.asr_model_name, target_device)
            self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.asr_model_name,
                map_location=map_loc,
            )
            self._asr_model.eval()
            self._asr_model = self._asr_model.to(map_loc)

    def apply_diarization_preset(
        self,
        preset_key: str = "offline_high_accuracy",
        chunk_len: int | None = None,
        chunk_right_context: int | None = None,
        fifo_len: int | None = None,
    ) -> None:
        """Configure Sortformer streaming/offline chunk parameters."""
        if self._diar_model is None:
            return
        preset = DIARIZATION_PRESETS.get(preset_key, DIARIZATION_PRESETS["offline_high_accuracy"])
        c_len = int(chunk_len if chunk_len is not None else preset["chunk_len"])
        c_right = int(
            chunk_right_context
            if chunk_right_context is not None
            else preset["chunk_right_context"]
        )
        f_len = int(fifo_len if fifo_len is not None else preset["fifo_len"])

        sortformer_modules = getattr(self._diar_model, "sortformer_modules", None)
        if sortformer_modules is not None:
            sortformer_modules.chunk_len = c_len
            sortformer_modules.chunk_right_context = c_right
            sortformer_modules.fifo_len = f_len

    def process_audio(
        self,
        audio_path: str | Path,
        preset_key: str = "offline_high_accuracy",
        nearest_tolerance: float = 1.5,
        merge_gap_threshold: float = 1.2,
    ) -> PipelineResult:
        """Run 16 kHz preprocessing, Nemotron-3-Diarization, Parakeet-TDT-0.6B-v3 ASR, and alignment."""
        start_wall = time.perf_counter()
        self.load_models()
        self.apply_diarization_preset(preset_key=preset_key)

        wav_16k_path, duration = preprocess_audio(audio_path)
        try:
            # 1. Run Speaker Diarization with nvidia/Nemotron-3-Diarization
            raw_diar = self._diar_model.diarize(audio=[wav_16k_path], batch_size=1)
            diar_segments = parse_diarization_segments(raw_diar)

            # 2. Run ASR with word-level timestamps using nvidia/parakeet-tdt-0.6b-v3
            asr_output = self._asr_model.transcribe(
                [wav_16k_path],
                batch_size=1,
                timestamps=True,
            )
            full_text, words = extract_word_timings(asr_output)

            # 3. Align words to speakers and construct speaker turns
            aligned_words, turns = align_words_with_speakers(
                words=words,
                diar_segments=diar_segments,
                nearest_tolerance=nearest_tolerance,
                merge_gap_threshold=merge_gap_threshold,
            )

            # If ASR returned only full text without timestamps, attach text to first turn
            if full_text and not aligned_words and turns:
                turns[0].text = full_text

            speaker_stats = compute_speaker_stats(diar_segments, turns)
            elapsed = time.perf_counter() - start_wall

            return PipelineResult(
                audio_duration=round(duration, 2),
                processing_time=round(elapsed, 2),
                full_text=full_text,
                diarization_segments=diar_segments,
                words=aligned_words,
                turns=turns,
                speaker_stats=speaker_stats,
                diarization_model=self.diarization_model_name,
                asr_model=self.asr_model_name,
            )
        finally:
            if os.path.exists(wav_16k_path):
                try:
                    os.remove(wav_16k_path)
                except OSError:
                    pass
