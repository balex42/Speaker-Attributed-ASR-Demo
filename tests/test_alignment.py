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
    clean_asr_text,
    compute_asr_chunk_windows,
    compute_speaker_stats,
    extract_word_timings,
    format_segments_as_rttm,
    format_turns_as_markdown,
    format_turns_as_srt,
    parse_diarization_segments,
    stitch_diarization_window,
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


def test_compute_asr_chunk_windows_splits_on_silence():
    diar_segs = [
        DiarizationSegment(start=0.0, end=12.0, speaker="Speaker 1"),
        DiarizationSegment(start=12.5, end=26.0, speaker="Speaker 2"),
        # Silence gap between 26.0s and 28.0s (ideal split point before 30s limit)
        DiarizationSegment(start=28.0, end=52.0, speaker="Speaker 1"),
        # Silence gap between 52.0s and 54.0s
        DiarizationSegment(start=54.0, end=65.0, speaker="Speaker 2"),
    ]
    windows = compute_asr_chunk_windows(audio_duration=65.0, diar_segments=diar_segs, max_chunk_sec=30.0)
    assert len(windows) == 3
    # First split should land inside the 26.0..28.0s silence gap (at 27.0s)
    assert windows[0] == (0.0, 27.0)
    # Second split should land inside the 52.0..54.0s silence gap (at 53.0s)
    assert windows[1] == (27.0, 53.0)
    assert windows[2] == (53.0, 65.0)


def test_strip_de_de_tags_and_sentence_boundary_smoothing():
    raw_hyp = SimpleNamespace(
        text="Die MDR jump fragt Fakt oder Fake. <de-DE> Ja, also in Belgien. <de-DE>",
        timestamp={
            "word": [
                {"word": "Die", "start": 1.36, "end": 1.60},
                {"word": "MDR", "start": 1.65, "end": 2.10},
                {"word": "jump", "start": 2.15, "end": 2.60},
                {"word": "fragt", "start": 3.00, "end": 3.80},
                {"word": "Fakt", "start": 5.20, "end": 5.60},
                {"word": "oder", "start": 5.70, "end": 6.16},
                # Delayed by RNNT lookahead slightly into Speaker 2's segment
                {"word": "Fake.", "start": 6.48, "end": 6.80},
                {"word": "<de-DE>", "start": 6.80, "end": 7.00},
                {"word": "Ja,", "start": 7.05, "end": 7.40},
                {"word": "also", "start": 7.45, "end": 7.90},
                {"word": "in", "start": 8.00, "end": 8.30},
                {"word": "Belgien.", "start": 8.35, "end": 9.10},
                {"word": "<de-DE>", "start": 9.10, "end": 9.30},
            ]
        },
    )
    full_text, words = extract_word_timings([raw_hyp])
    assert "<de-DE>" not in full_text
    assert all("<de-DE>" not in w.word for w in words)

    diar_segs = [
        DiarizationSegment(start=1.20, end=6.30, speaker="Speaker 1"),
        DiarizationSegment(start=6.35, end=10.00, speaker="Speaker 2"),
    ]
    _, turns = align_words_with_speakers(words, diar_segs)
    assert len(turns) == 2
    assert turns[0].speaker == "Speaker 1"
    assert turns[0].text == "Die MDR jump fragt Fakt oder Fake."
    assert turns[1].speaker == "Speaker 2"
    assert turns[1].text == "Ja, also in Belgien."


def test_stitch_diarization_window_speaker_permutation():
    prev_segs = [
        DiarizationSegment(start=0.0, end=250.0, speaker="Speaker 1"),
        DiarizationSegment(start=255.0, end=480.0, speaker="Speaker 2"),
    ]
    # Second window starts at 460.0s (20s overlap: 460..480s).
    # In the new window, local "Speaker 1" talks in 0..20s (460..480s), which corresponds to global "Speaker 2"!
    new_window_segs = [
        DiarizationSegment(start=0.0, end=50.0, speaker="Speaker 1"),
        DiarizationSegment(start=55.0, end=120.0, speaker="Speaker 2"),
    ]
    merged = stitch_diarization_window(
        existing_segments=prev_segs,
        new_segments=new_window_segs,
        win_start=460.0,
        overlap_sec=20.0,
    )
    # Local Speaker 1 should be remapped to global Speaker 2 and stitched across the 470s midpoint
    assert merged[1].speaker == "Speaker 2"
    assert merged[2].speaker == "Speaker 2"
    assert merged[2].start == 470.0
    assert merged[2].end == 510.0
    # Local Speaker 2 should be remapped to global Speaker 1
    assert merged[3].speaker == "Speaker 1"


if __name__ == "__main__":
    test_parse_diarization_segments_formats()
    test_extract_word_timings_from_parakeet_hypothesis()
    test_align_words_with_speakers_and_overlap()
    test_exports_and_visual_report()
    test_compute_asr_chunk_windows_splits_on_silence()
    test_strip_de_de_tags_and_sentence_boundary_smoothing()
    test_stitch_diarization_window_speaker_permutation()
    print("All alignment and formatting tests passed!")

