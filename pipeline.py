"""Core pipeline combining NVIDIA Nemotron-3-Diarization and Canary-1B-v2.

- Diarization: `nvidia/Nemotron-3-Diarization` (`SortformerEncLabelModel`)
- ASR + Word Timestamps: `nvidia/canary-1b-v2` (`ASRModel` / `EncDecMultiTaskModel`)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import gc
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

ProgressCallback = Callable[[float, str], None]

logger = logging.getLogger(__name__)

DIARIZATION_MODEL_ID = "nvidia/Nemotron-3-Diarization"
ASR_MODEL_ID = "nvidia/canary-1b-v2"
TARGET_SAMPLE_RATE = 16000

# Supported 25 European languages for nvidia/canary-1b-v2
SUPPORTED_ASR_LANGUAGES: list[tuple[str, str]] = [
    ("German (de)", "de"),
    ("English (en)", "en"),
    ("French (fr)", "fr"),
    ("Spanish (es)", "es"),
    ("Italian (it)", "it"),
    ("Dutch (nl)", "nl"),
    ("Polish (pl)", "pl"),
    ("Portuguese (pt)", "pt"),
    ("Danish (da)", "da"),
    ("Swedish (sv)", "sv"),
    ("Finnish (fi)", "fi"),
    ("Czech (cs)", "cs"),
    ("Slovak (sk)", "sk"),
    ("Slovenian (sl)", "sl"),
    ("Croatian (hr)", "hr"),
    ("Bulgarian (bg)", "bg"),
    ("Romanian (ro)", "ro"),
    ("Hungarian (hu)", "hu"),
    ("Greek (el)", "el"),
    ("Estonian (et)", "et"),
    ("Latvian (lv)", "lv"),
    ("Lithuanian (lt)", "lt"),
    ("Maltese (mt)", "mt"),
    ("Russian (ru)", "ru"),
    ("Ukrainian (uk)", "uk"),
]

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


def stitch_diarization_window(
    existing_segments: list[DiarizationSegment],
    new_segments: list[DiarizationSegment],
    win_start: float,
    overlap_sec: float,
) -> list[DiarizationSegment]:
    """Map speaker labels in `new_segments` (offset by `win_start`) to match `existing_segments` across `[win_start, win_start + overlap_sec]`."""
    offset_new = [
        DiarizationSegment(
            start=round(s.start + win_start, 3),
            end=round(s.end + win_start, 3),
            speaker=s.speaker,
        )
        for s in new_segments
    ]
    if not existing_segments or win_start <= 0.0:
        return offset_new

    overlap_end = win_start + overlap_sec
    midpoint = win_start + 0.5 * overlap_sec

    # Compute pairwise temporal overlap between existing speakers and new window speakers inside [win_start, overlap_end]
    cooccurrence: dict[tuple[str, str], float] = {}
    existing_in_zone = [s for s in existing_segments if s.end > win_start and s.start < overlap_end]
    new_in_zone = [s for s in offset_new if s.end > win_start and s.start < overlap_end]

    for es in existing_in_zone:
        for ns in new_in_zone:
            ov = min(es.end, ns.end, overlap_end) - max(es.start, ns.start, win_start)
            if ov > 0:
                key = (ns.speaker, es.speaker)
                cooccurrence[key] = cooccurrence.get(key, 0.0) + ov

    # Greedy 1-to-1 matching by highest overlap duration
    speaker_map: dict[str, str] = {}
    used_existing: set[str] = set()
    for (new_spk, ex_spk), _ in sorted(cooccurrence.items(), key=lambda kv: -kv[1]):
        if new_spk not in speaker_map and ex_spk not in used_existing:
            speaker_map[new_spk] = ex_spk
            used_existing.add(ex_spk)

    # Ensure any unmatched speaker in new_segments does not collide with an already-assigned existing speaker
    all_existing_spks = list(dict.fromkeys(s.speaker for s in existing_segments))
    unused_existing = [spk for spk in all_existing_spks if spk not in used_existing]
    for ns in offset_new:
        if ns.speaker not in speaker_map:
            if ns.speaker not in used_existing:
                speaker_map[ns.speaker] = ns.speaker
                used_existing.add(ns.speaker)
            elif unused_existing:
                assigned = unused_existing.pop(0)
                speaker_map[ns.speaker] = assigned
                used_existing.add(assigned)
            else:
                idx = len(used_existing) + 1
                while f"Speaker {idx}" in used_existing:
                    idx += 1
                assigned = f"Speaker {idx}"
                speaker_map[ns.speaker] = assigned
                used_existing.add(assigned)

    # Keep existing segments up to `midpoint` and stitched new segments from `midpoint` onward
    merged: list[DiarizationSegment] = []
    for es in existing_segments:
        if es.end <= midpoint:
            merged.append(es)
        elif es.start < midpoint:
            merged.append(DiarizationSegment(start=es.start, end=round(midpoint, 3), speaker=es.speaker))

    for ns in offset_new:
        mapped_spk = speaker_map.get(ns.speaker, ns.speaker)
        if ns.start >= midpoint:
            merged.append(DiarizationSegment(start=ns.start, end=ns.end, speaker=mapped_spk))
        elif ns.end > midpoint:
            merged.append(
                DiarizationSegment(start=round(midpoint, 3), end=ns.end, speaker=mapped_spk)
            )

    merged.sort(key=lambda s: (s.start, s.end, s.speaker))
    return merged


_LANG_TAG_RE = re.compile(
    r"<(?:lang-[^>]+|\|[^|]+\||/?[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})+|[a-z]{2})>"
)
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]»›]*$")


def clean_asr_text(text: str) -> str:
    """Strip inline language tags such as `<de-DE>`, `<en-US>`, or `<lang-de-DE>` emitted by Nemotron 3.5 ASR."""
    return re.sub(r"\s+", " ", _LANG_TAG_RE.sub(" ", text)).strip()


def synthesize_word_timings(text: str, duration: float) -> list[WordTiming]:
    """Interpolate word-level timestamps across `[0, duration]` weighted by character length."""
    cleaned = clean_asr_text(text)
    tokens = [t for t in cleaned.split(" ") if t]
    if not tokens:
        return []
    duration = max(0.05, float(duration))
    weights = [max(1, len(t)) for t in tokens]
    total_weight = float(sum(weights))
    cursor = 0.0
    result: list[WordTiming] = []
    for tok, w in zip(tokens, weights):
        span = duration * (float(w) / total_weight)
        w_start = round(cursor, 3)
        w_end = round(min(duration, cursor + span), 3)
        result.append(WordTiming(word=tok, start=w_start, end=max(w_start, w_end)))
        cursor += span
    return result


def extract_word_timings(
    asr_hypothesis: Any, fallback_duration: float = 0.0
) -> tuple[str, list[WordTiming]]:
    """Extract full text and word-level (or segment/interpolated fallback) timestamps from ASR output."""
    if isinstance(asr_hypothesis, list) and len(asr_hypothesis) > 0:
        asr_hypothesis = asr_hypothesis[0]

    if isinstance(asr_hypothesis, str):
        cleaned = clean_asr_text(asr_hypothesis)
        words = synthesize_word_timings(cleaned, fallback_duration) if fallback_duration > 0 else []
        return cleaned, words

    raw_text = getattr(asr_hypothesis, "text", "") or ""
    full_text = clean_asr_text(raw_text)
    timestamp_dict = getattr(asr_hypothesis, "timestamp", None) or getattr(
        asr_hypothesis, "timestep", None
    )

    words: list[WordTiming] = []
    if isinstance(timestamp_dict, dict):
        raw_words = timestamp_dict.get("word") or []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            word_str = clean_asr_text(str(item.get("word", item.get("char", ""))))
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
                seg_text = clean_asr_text(str(seg.get("segment", seg.get("text", ""))))
                if not seg_text:
                    continue
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start))
                words.append(WordTiming(word=seg_text, start=start, end=max(start, end)))

    if not words and full_text and fallback_duration > 0:
        words = synthesize_word_timings(full_text, fallback_duration)

    return full_text, words


def compute_asr_chunk_windows(
    audio_duration: float,
    diar_segments: list[DiarizationSegment],
    max_chunk_sec: float = 30.0,
    min_chunk_sec: float = 10.0,
) -> list[tuple[float, float]]:
    """Split `[0, audio_duration]` into chunk windows of at most `max_chunk_sec` seconds.

    Prefers splitting at the midpoint of silence gaps between `diar_segments` so words
    are not cut mid-utterance. Falls back to segment boundaries or `max_chunk_sec`.
    """
    audio_duration = max(0.0, float(audio_duration))
    if max_chunk_sec <= 0 or audio_duration <= max_chunk_sec:
        return [(0.0, round(audio_duration, 3))]

    # Merge overlapping diarization intervals into contiguous active-speech spans
    sorted_segs = sorted(diar_segments, key=lambda s: (s.start, s.end))
    speech_spans: list[ list[float] ] = []
    for seg in sorted_segs:
        s_start = max(0.0, min(audio_duration, seg.start))
        s_end = max(s_start, min(audio_duration, seg.end))
        if s_end <= s_start:
            continue
        if speech_spans and s_start <= speech_spans[-1][1] + 0.02:
            speech_spans[-1][1] = max(speech_spans[-1][1], s_end)
        else:
            speech_spans.append([s_start, s_end])

    # Compute silence gaps as the complement of speech_spans in [0, audio_duration]
    silence_gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for s_start, s_end in speech_spans:
        if s_start - prev_end >= 0.05:
            silence_gaps.append((prev_end, s_start))
        prev_end = max(prev_end, s_end)
    if audio_duration - prev_end >= 0.05:
        silence_gaps.append((prev_end, audio_duration))

    # Collect raw segment boundary timestamps as secondary split candidates
    seg_boundaries = sorted(
        {
            round(t, 3)
            for seg in sorted_segs
            for t in (seg.start, seg.end)
            if 0.0 < t < audio_duration
        }
    )

    effective_min = min(min_chunk_sec, max_chunk_sec * 0.5)
    windows: list[tuple[float, float]] = []
    cursor = 0.0

    while (audio_duration - cursor) > max_chunk_sec:
        search_start = cursor + effective_min
        search_end = cursor + max_chunk_sec
        split_point: float | None = None

        # 1. Prefer the widest silence gap inside [search_start, search_end]
        best_gap_score = -1.0
        for g_start, g_end in silence_gaps:
            c_start = max(g_start, search_start)
            c_end = min(g_end, search_end)
            gap_len = c_end - c_start
            if gap_len >= 0.04:
                # Score favors wider gaps and splits closer to search_end
                score = gap_len + 0.1 * ((c_end - search_start) / max(1e-3, max_chunk_sec))
                if score > best_gap_score:
                    best_gap_score = score
                    split_point = 0.5 * (c_start + c_end)

        # 2. Fallback: speaker segment boundary inside [search_start, search_end]
        if split_point is None:
            candidates = [b for b in seg_boundaries if search_start <= b <= search_end]
            if candidates:
                split_point = candidates[-1]

        # 3. Fallback: hard cut at search_end
        if split_point is None or split_point <= cursor:
            split_point = search_end

        windows.append((round(cursor, 3), round(split_point, 3)))
        cursor = split_point

    if audio_duration - cursor > 1e-3:
        # If the final tail is very short (< 1.5s), merge it into the previous window
        if windows and (audio_duration - cursor) < 1.5:
            prev_start, _ = windows[-1]
            windows[-1] = (prev_start, round(audio_duration, 3))
        else:
            windows.append((round(cursor, 3), round(audio_duration, 3)))

    return windows


def _smooth_sentence_speaker_boundaries(
    words: list[WordTiming],
    max_orphan_words: int = 2,
    max_boundary_gap: float = 1.0,
) -> None:
    """Snap 1-2 word sentence-boundary orphans caused by RNNT lookahead lag back to their sentence's speaker.

    - Trailing orphans: If the last 1-2 words of a sentence (ending in '.', '!', '?') were pushed into
      the next speaker's segment while the preceding words of the sentence belong to the previous speaker,
      reassign those trailing 1-2 words to the sentence's speaker.
    - Leading orphans: If the first 1-2 words of a new sentence were assigned to the previous speaker while
      the body of the new sentence belongs to the next speaker, reassign those leading 1-2 words forward.
    - Mid-sentence 1-word islands: If a single word mid-sentence is flanked by the same speaker on both sides,
      reassign it to the surrounding speaker.
    """
    n = len(words)
    if n < 3:
        return

    # 1. Split word indices into sentences based on terminal punctuation (., !, ?)
    sentences: list[list[int]] = []
    current_sent: list[int] = []
    for idx, w in enumerate(words):
        current_sent.append(idx)
        if _SENTENCE_END_RE.search(w.word):
            sentences.append(current_sent)
            current_sent = []
    if current_sent:
        sentences.append(current_sent)

    for sent_indices in sentences:
        m = len(sent_indices)
        if m < 3:
            continue

        # Smooth single-word mid-sentence islands first
        for k in range(1, m - 1):
            prev_w = words[sent_indices[k - 1]]
            curr_w = words[sent_indices[k]]
            next_w = words[sent_indices[k + 1]]
            if (
                prev_w.speaker == next_w.speaker
                and curr_w.speaker != prev_w.speaker
                and (curr_w.start - prev_w.end) <= max_boundary_gap
                and (next_w.start - curr_w.end) <= max_boundary_gap
            ):
                curr_w.speaker = prev_w.speaker

        # Check trailing 1..max_orphan_words at the end of the sentence
        for tail_len in range(1, min(max_orphan_words, m - 1) + 1):
            body_end_idx = sent_indices[m - tail_len - 1]
            tail_start_idx = sent_indices[m - tail_len]
            tail_indices = sent_indices[m - tail_len :]
            tail_speakers = {words[i].speaker for i in tail_indices}
            body_speaker = words[body_end_idx].speaker

            # Require at least 2 words before the tail with a consistent speaker (or whole body if m - tail_len == 1)
            prev_body_idx = sent_indices[max(0, m - tail_len - 2)]
            if (
                len(tail_speakers) == 1
                and body_speaker not in tail_speakers
                and words[prev_body_idx].speaker == body_speaker
                and (words[tail_start_idx].start - words[body_end_idx].end) <= max_boundary_gap
            ):
                for i in tail_indices:
                    words[i].speaker = body_speaker
                break

        # Check leading 1..max_orphan_words at the start of the sentence
        for head_len in range(1, min(max_orphan_words, m - 2) + 1):
            head_indices = sent_indices[:head_len]
            body_start_idx = sent_indices[head_len]
            next_body_idx = sent_indices[min(m - 1, head_len + 1)]
            head_speakers = {words[i].speaker for i in head_indices}
            body_speaker = words[body_start_idx].speaker

            if (
                len(head_speakers) == 1
                and body_speaker not in head_speakers
                and words[next_body_idx].speaker == body_speaker
                and (words[body_start_idx].start - words[head_indices[-1]].end) <= max_boundary_gap
            ):
                for i in head_indices:
                    words[i].speaker = body_speaker
                break


def align_words_with_speakers(
    words: list[WordTiming],
    diar_segments: list[DiarizationSegment],
    nearest_tolerance: float = 1.5,
    merge_gap_threshold: float = 4.0,
    asr_timestamp_shift: float = 0.0,
) -> tuple[list[WordTiming], list[SpeakerTurn]]:
    """Align word-level ASR timestamps with speaker diarization intervals.

    1. Applies `asr_timestamp_shift` (default -0.20s) when computing overlap with diarization
       segments to compensate for streaming RNNT token emission lag.
    2. For each word `[w_start, w_end]`, computes overlap duration with every active
       `DiarizationSegment` and assigns the speaker with the highest temporal overlap.
    3. If a word has zero direct overlap with any segment, assigns the temporally closest
       diarization segment within `nearest_tolerance` seconds.
    4. Applies sentence-boundary speaker smoothing so RNNT lookahead lag at turn boundaries
       does not split the last/first 1-2 words of a sentence onto the adjacent speaker.
    5. Groups consecutive words assigned to the same speaker into `SpeakerTurn` objects.
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
        # Shift word window slightly earlier for overlap matching to compensate for RNNT emission lag
        w_start = max(0.0, w.start + asr_timestamp_shift)
        w_end = max(w_start + eps, w.end + asr_timestamp_shift)
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

    # Smooth 1-2 word sentence-boundary orphans caused by RNNT emission lag
    _smooth_sentence_speaker_boundaries(aligned_words)

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
    """Convert any audio file to 16 kHz mono WAV required by Nemotron-3-Diarization & Canary.

    Streams conversion via `ffmpeg` on disk so multi-hour audio files do not spike RAM,
    falling back to `librosa` only if `ffmpeg` is unavailable.
    Returns `(wav_path, duration_seconds)`.
    """
    import soundfile as sf

    tmp_fd, tmp_wav_path = tempfile.mkstemp(suffix="_16k_mono.wav", prefix="nemo_demo_")
    os.close(tmp_fd)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-acodec",
            "pcm_s16le",
            tmp_wav_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(tmp_wav_path) and os.path.getsize(tmp_wav_path) > 44:
            info = sf.info(tmp_wav_path)
            return tmp_wav_path, float(info.duration)
        logger.warning(
            "ffmpeg conversion exited with %s, falling back to librosa: %s",
            proc.returncode,
            proc.stderr.decode("utf-8", errors="ignore")[-300:],
        )

    import librosa
    import numpy as np

    audio_data, _ = librosa.load(str(input_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)

    duration = float(len(audio_data)) / float(TARGET_SAMPLE_RATE)
    sf.write(tmp_wav_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
    del audio_data
    gc.collect()
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

    def load_models(self, progress_callback: ProgressCallback | None = None) -> None:
        """Lazily load both NeMo models onto the target device."""
        if progress_callback and (self._diar_model is None or self._asr_model is None):
            progress_callback(0.05, "Step 1/6: Initializing NVIDIA NeMo framework...")

        import torch
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models import SortformerEncLabelModel

        target_device = self._resolve_device()
        map_loc = torch.device(target_device)

        if self._diar_model is None:
            if progress_callback:
                progress_callback(
                    0.15,
                    f"Step 2/6: Loading diarization model ({self.diarization_model_name}) on {target_device}...",
                )
            logger.info("Loading diarization model: %s on %s", self.diarization_model_name, target_device)
            self._diar_model = SortformerEncLabelModel.from_pretrained(
                self.diarization_model_name,
                map_location=map_loc,
            )
            self._diar_model.eval()
            self._diar_model = self._diar_model.to(map_loc)

        if self._asr_model is None:
            if progress_callback:
                progress_callback(
                    0.35,
                    f"Step 3/6: Loading ASR model ({self.asr_model_name}) on {target_device}...",
                )
            logger.info("Loading ASR model: %s on %s", self.asr_model_name, target_device)
            self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.asr_model_name,
                map_location=map_loc,
            )
            self._asr_model.eval()
            self._asr_model = self._asr_model.to(map_loc)
            gc.collect()

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

    def _transcribe_batch_wavs(
        self,
        wav_paths: list[str],
        durations: list[float],
        language: str = "de",
        batch_size: int = 1,
    ) -> list[tuple[str, list[WordTiming]]]:
        """Run Canary-1B-v2 transcription on a batch of WAV files with locked source/target language and word timestamps."""
        if not wav_paths:
            return []
        lang_code = (language or "de").strip().split("-")[0].lower() or "de"
        eff_batch = max(1, min(int(batch_size), len(wav_paths)))

        try:
            asr_outputs = self._asr_model.transcribe(
                wav_paths,
                batch_size=eff_batch,
                source_lang=lang_code,
                target_lang=lang_code,
                timestamps=True,
                pnc="yes",
            )
        except TypeError:
            asr_outputs = self._asr_model.transcribe(
                wav_paths,
                batch_size=eff_batch,
                source_lang=lang_code,
                target_lang=lang_code,
                timestamps=True,
            )

        if not isinstance(asr_outputs, list):
            asr_outputs = [asr_outputs]

        results: list[tuple[str, list[WordTiming]]] = []
        for idx, dur in enumerate(durations):
            hyp = asr_outputs[idx] if idx < len(asr_outputs) else ""
            results.append(extract_word_timings(hyp, fallback_duration=dur))
        return results

    def _transcribe_audio_chunks(
        self,
        wav_16k_path: str | Path,
        audio_duration: float,
        diar_segments: list[DiarizationSegment],
        language: str = "de",
        max_chunk_sec: float = 30.0,
        batch_size: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[str, list[WordTiming]]:
        """Transcribe audio in silence-aligned chunks (with configurable mini-batch size) using disk streaming."""
        import soundfile as sf

        batch_size = max(1, int(batch_size))
        windows = compute_asr_chunk_windows(
            audio_duration=audio_duration,
            diar_segments=diar_segments,
            max_chunk_sec=max_chunk_sec,
        )
        num_chunks = len(windows)

        # Single-chunk fast path
        if num_chunks <= 1:
            if progress_callback:
                progress_callback(
                    0.65,
                    f"Step 6/6: Transcribing audio ({format_timestamp(0.0)} - {format_timestamp(audio_duration)}, lang={language}) with {self.asr_model_name}...",
                )
            batch_res = self._transcribe_batch_wavs(
                wav_paths=[str(wav_16k_path)],
                durations=[audio_duration],
                language=language,
                batch_size=1,
            )
            return batch_res[0] if batch_res else ("", [])

        all_texts: list[str] = []
        all_words: list[WordTiming] = []

        # Stream slices from disk via SoundFile.seek() so we never load the full WAV into RAM
        with sf.SoundFile(str(wav_16k_path), "r") as snd:
            sr = snd.samplerate
            total_frames = len(snd)

            for batch_start in range(0, num_chunks, batch_size):
                batch_windows = windows[batch_start : batch_start + batch_size]
                batch_end_idx = batch_start + len(batch_windows)

                if progress_callback:
                    frac = 0.62 + 0.32 * (batch_start / num_chunks)
                    w0_start = batch_windows[0][0]
                    w1_end = batch_windows[-1][1]
                    if len(batch_windows) == 1:
                        chunk_label = f"chunk {batch_start + 1}/{num_chunks}"
                    else:
                        chunk_label = f"chunks {batch_start + 1}-{batch_end_idx}/{num_chunks} (batch={len(batch_windows)})"
                    progress_callback(
                        frac,
                        f"Step 6/6: Canary-1B-v2 ASR {chunk_label} "
                        f"({format_timestamp(w0_start)} - {format_timestamp(w1_end)}, lang={language})...",
                    )

                batch_wav_paths: list[str] = []
                batch_durations: list[float] = []
                batch_offsets: list[float] = []

                try:
                    for local_idx, (win_start, win_end) in enumerate(batch_windows):
                        start_frame = max(0, min(total_frames, int(round(win_start * sr))))
                        end_frame = max(start_frame, min(total_frames, int(round(win_end * sr))))
                        num_frames = end_frame - start_frame
                        if num_frames < int(0.05 * sr):
                            continue

                        snd.seek(start_frame)
                        chunk_samples = snd.read(num_frames, dtype="int16")
                        chunk_dur = float(len(chunk_samples)) / float(sr)

                        tmp_fd, chunk_wav_path = tempfile.mkstemp(
                            suffix=f"_chunk_{batch_start + local_idx + 1}.wav",
                            prefix="nemo_chunk_",
                        )
                        os.close(tmp_fd)
                        sf.write(chunk_wav_path, chunk_samples, sr, subtype="PCM_16")
                        del chunk_samples

                        batch_wav_paths.append(chunk_wav_path)
                        batch_durations.append(chunk_dur)
                        batch_offsets.append(win_start)

                    if batch_wav_paths:
                        batch_results = self._transcribe_batch_wavs(
                            wav_paths=batch_wav_paths,
                            durations=batch_durations,
                            language=language,
                            batch_size=batch_size,
                        )
                        for win_start, (chunk_text, chunk_words) in zip(batch_offsets, batch_results):
                            if chunk_text.strip():
                                all_texts.append(chunk_text.strip())
                            for w in chunk_words:
                                all_words.append(
                                    WordTiming(
                                        word=w.word,
                                        start=round(w.start + win_start, 3),
                                        end=round(w.end + win_start, 3),
                                    )
                                )
                finally:
                    for p in batch_wav_paths:
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    gc.collect()

        return " ".join(all_texts), all_words

    def _diarize_audio_streaming_safe(
        self,
        wav_16k_path: str | Path,
        audio_duration: float,
        max_window_sec: float = 480.0,
        overlap_sec: float = 20.0,
        progress_callback: ProgressCallback | None = None,
    ) -> list[DiarizationSegment]:
        """Run Nemotron-3-Diarization in bounded windows so CPU flex_attention never OOMs on hour-long audio."""
        import soundfile as sf

        if audio_duration <= max_window_sec:
            if progress_callback:
                progress_callback(
                    0.55,
                    f"Step 5/6: Running speaker diarization ({self.diarization_model_name})...",
                )
            raw_diar = self._diar_model.diarize(audio=[str(wav_16k_path)], batch_size=1)
            segs = parse_diarization_segments(raw_diar)
            del raw_diar
            gc.collect()
            return segs

        step_sec = max(60.0, max_window_sec - overlap_sec)
        windows: list[tuple[float, float]] = []
        cursor = 0.0
        while cursor < audio_duration:
            win_end = min(audio_duration, cursor + max_window_sec)
            windows.append((round(cursor, 3), round(win_end, 3)))
            if win_end >= audio_duration:
                break
            cursor += step_sec

        stitched: list[DiarizationSegment] = []
        with sf.SoundFile(str(wav_16k_path), "r") as snd:
            sr = snd.samplerate
            total_frames = len(snd)

            for idx, (win_start, win_end) in enumerate(windows):
                if progress_callback:
                    frac = 0.52 + 0.10 * (idx / len(windows))
                    progress_callback(
                        frac,
                        f"Step 5/6: Diarizing window {idx + 1}/{len(windows)} "
                        f"({format_timestamp(win_start)} - {format_timestamp(win_end)})...",
                    )

                start_frame = max(0, min(total_frames, int(round(win_start * sr))))
                end_frame = max(start_frame, min(total_frames, int(round(win_end * sr))))
                snd.seek(start_frame)
                win_samples = snd.read(end_frame - start_frame, dtype="int16")

                tmp_fd, win_wav_path = tempfile.mkstemp(
                    suffix=f"_diar_{idx + 1}.wav", prefix="nemo_diar_"
                )
                os.close(tmp_fd)
                try:
                    sf.write(win_wav_path, win_samples, sr, subtype="PCM_16")
                    del win_samples
                    raw_diar = self._diar_model.diarize(audio=[win_wav_path], batch_size=1)
                    win_segs = parse_diarization_segments(raw_diar)
                    del raw_diar
                    stitched = stitch_diarization_window(
                        existing_segments=stitched,
                        new_segments=win_segs,
                        win_start=win_start,
                        overlap_sec=overlap_sec,
                    )
                finally:
                    if os.path.exists(win_wav_path):
                        try:
                            os.remove(win_wav_path)
                        except OSError:
                            pass
                    gc.collect()

        return stitched

    def process_audio(
        self,
        audio_path: str | Path,
        preset_key: str = "offline_high_accuracy",
        language: str = "de",
        nearest_tolerance: float = 1.5,
        merge_gap_threshold: float = 4.0,
        max_chunk_sec: float = 30.0,
        batch_size: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Run 16 kHz preprocessing, Nemotron-3-Diarization, Canary-1B-v2 ASR, and alignment."""
        import torch

        start_wall = time.perf_counter()
        self.load_models(progress_callback=progress_callback)
        self.apply_diarization_preset(preset_key=preset_key)

        if progress_callback:
            progress_callback(0.50, "Step 4/6: Preprocessing & resampling audio to 16 kHz mono WAV...")
        wav_16k_path, duration = preprocess_audio(audio_path)
        try:
            with torch.inference_mode():
                # 1. Run Speaker Diarization with nvidia/Nemotron-3-Diarization (windowed for long files)
                diar_segments = self._diarize_audio_streaming_safe(
                    wav_16k_path=wav_16k_path,
                    audio_duration=duration,
                    progress_callback=progress_callback,
                )

                # 2. Run ASR with language conditioning using nvidia/canary-1b-v2
                full_text, words = self._transcribe_audio_chunks(
                    wav_16k_path=wav_16k_path,
                    audio_duration=duration,
                    diar_segments=diar_segments,
                    language=language,
                    max_chunk_sec=max_chunk_sec,
                    batch_size=batch_size,
                    progress_callback=progress_callback,
                )
                gc.collect()

            # 3. Align words to speakers and construct speaker turns
            if progress_callback:
                progress_callback(0.95, "Aligning word timestamps with speaker turns...")
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

            if progress_callback:
                progress_callback(1.0, "Complete!")

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
