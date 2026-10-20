"""Unit tests for Nemotron-3-Diarization + Parakeet-TDT-0.6B-v3 alignment and formatting."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import render_visual_report
from pipeline import (
    DiarizationSegment,
    PipelineResult,
    WordTiming,
    align_words_with_speakers,
    compute_speaker_stats,
    extract_word_timings,
    format_segments_as_rttm,
    format_turns_as_markdown,
    format_turns_as_srt,
    parse_diarization_segments,
)


def test_parse_diarization_segments_formats():
    raw = [
        [
            "0.120, 1.850, speaker_0",
            "1.700 3.400 speaker_1",
            "SPEAKER sample 1 3.500 1.500 <NA> <NA> speaker_0 <NA> <NA>",
        ]
    ]
    segs = parse_diarization_segments(raw)
    assert len(segs) == 3
    assert segs[0] == DiarizationSegment(start=0.12, end=1.85, speaker="Speaker 1")
    assert segs[1] == DiarizationSegment(start=1.70, end=3.40, speaker="Speaker 2")
    assert segs[2] == DiarizationSegment(start=3.50, end=5.00, speaker="Speaker 1")


def test_extract_word_timings_from_parakeet_hypothesis():
    mock_hyp = SimpleNamespace(
        text="Hello there. General Kenobi.",
        timestamp={
            "word": [
                {"word": "Hello", "start": 0.20, "end": 0.55},
                {"word": "there.", "start": 0.60, "end": 1.10},
                {"word": "General", "start": 1.80, "end": 2.30},
                {"word": "Kenobi.", "start": 2.35, "end": 3.00},
            ]
        },
    )
    text, words = extract_word_timings([mock_hyp])
    assert text == "Hello there. General Kenobi."
    assert len(words) == 4
    assert words[0].word == "Hello"
    assert words[2].start == 1.80


def test_align_words_with_speakers_and_overlap():
    diar_segs = [
        DiarizationSegment(start=0.10, end=1.50, speaker="Speaker 1"),
        DiarizationSegment(start=1.30, end=3.20, speaker="Speaker 2"),
    ]
    words = [
        WordTiming(word="Hello", start=0.20, end=0.55),
        WordTiming(word="there.", start=0.60, end=1.10),
        # Partially overlapping word (1.32 - 1.48 overlaps both Speaker 1 & Speaker 2, more with Speaker 2 if 1.35..1.80)
        WordTiming(word="General", start=1.35, end=2.10),
        WordTiming(word="Kenobi.", start=2.20, end=2.95),
    ]
    aligned_words, turns = align_words_with_speakers(words, diar_segs)

    assert len(turns) == 2
    assert turns[0].speaker == "Speaker 1"
    assert turns[0].text == "Hello there."
    assert turns[1].speaker == "Speaker 2"
    assert turns[1].text == "General Kenobi."
    assert turns[1].has_overlap is True
    assert set(aligned_words[2].overlapping_speakers) == {"Speaker 1", "Speaker 2"}


def test_exports_and_visual_report():
    diar_segs = [
        DiarizationSegment(start=0.0, end=2.0, speaker="Speaker 1"),
        DiarizationSegment(start=2.2, end=4.5, speaker="Speaker 2"),
    ]
    words = [
        WordTiming(word="Good", start=0.2, end=0.6),
        WordTiming(word="morning!", start=0.7, end=1.5),
        WordTiming(word="Welcome", start=2.3, end=3.0),
        WordTiming(word="aboard.", start=3.1, end=4.1),
    ]
    aligned_words, turns = align_words_with_speakers(words, diar_segs)
    stats = compute_speaker_stats(diar_segs, turns)

    result = PipelineResult(
        audio_duration=4.5,
        processing_time=0.45,
        full_text="Good morning! Welcome aboard.",
        diarization_segments=diar_segs,
        words=aligned_words,
        turns=turns,
        speaker_stats=stats,
    )

    md = format_turns_as_markdown(turns)
    srt = format_turns_as_srt(turns)
    rttm = format_segments_as_rttm(diar_segs, file_id="meeting_clip")
    html_report = render_visual_report(result)

    assert "**Speaker 1**" in md and "Good morning!" in md
    assert "[Speaker 2] Welcome aboard." in srt
    assert "SPEAKER meeting_clip 1 0.000 2.000 <NA> <NA> speaker_1 <NA> <NA>" in rttm
    assert "Speaker Activity Timeline" in html_report
    assert len(stats) == 2


if __name__ == "__main__":
    test_parse_diarization_segments_formats()
    test_extract_word_timings_from_parakeet_hypothesis()
    test_align_words_with_speakers_and_overlap()
    test_exports_and_visual_report()
    print("All alignment and formatting tests passed!")

