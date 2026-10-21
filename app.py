"""Interactive Web UI & CLI demo for NVIDIA Nemotron-3-Diarization + Parakeet-TDT-0.6B-v3."""

from __future__ import annotations

import argparse
import html
import logging
import os
import tempfile
from pathlib import Path

from pipeline import (
    ASR_MODEL_ID,
    DIARIZATION_MODEL_ID,
    DIARIZATION_PRESETS,
    SUPPORTED_ASR_LANGUAGES,
    DiarizationASRPipeline,
    PipelineResult,
    format_segments_as_rttm,
    format_timestamp,
    format_turns_as_markdown,
    format_turns_as_srt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nemo_demo")

# Distinct color palette for up to 8 speakers supported by Nemotron-3-Diarization
SPEAKER_PALETTE: list[tuple[str, str, str]] = [
    ("#76b900", "#eef8d6", "#2c4700"),  # Speaker 1: NVIDIA Green
    ("#2563eb", "#dbeafe", "#1e3a8a"),  # Speaker 2: Royal Blue
    ("#d97706", "#fef3c7", "#78350f"),  # Speaker 3: Amber
    ("#9333ea", "#f3e8ff", "#581c87"),  # Speaker 4: Purple
    ("#e11d48", "#ffe4e6", "#881337"),  # Speaker 5: Rose
    ("#0d9488", "#ccfbf1", "#134e4a"),  # Speaker 6: Teal
    ("#4f46e5", "#e0e7ff", "#312e81"),  # Speaker 7: Indigo
    ("#ea580c", "#ffedd5", "#7c2d12"),  # Speaker 8: Orange
]


def _speaker_colors(speaker: str) -> tuple[str, str, str]:
    """Return `(accent_hex, bg_hex, text_hex)` for a speaker label."""
    digits = "".join(ch for ch in speaker if ch.isdigit())
    idx = (int(digits) - 1) if digits else hash(speaker)
    return SPEAKER_PALETTE[idx % len(SPEAKER_PALETTE)]


def render_visual_report(result: PipelineResult) -> str:
    """Render an HTML speaker timeline swimlane chart + color-coded conversation view."""
    total_dur = max(result.audio_duration, 0.01)
    speakers = [s.speaker for s in result.speaker_stats]

    # 1. Summary Metrics Bar
    rtf = result.processing_time / total_dur if total_dur > 0 else 0.0
    summary_html = f"""
    <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px;font-family:system-ui,-apple-system,sans-serif;">
      <div style="flex:1;min-width:140px;background:#f9fafb;border:1px solid #e5e7eb;padding:12px 16px;border-radius:10px;">
        <div style="font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.04em;">Audio Duration</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px;color:#111827;">{format_timestamp(result.audio_duration)} ({result.audio_duration:.1f}s)</div>
      </div>
      <div style="flex:1;min-width:140px;background:#f9fafb;border:1px solid #e5e7eb;padding:12px 16px;border-radius:10px;">
        <div style="font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.04em;">Speakers Detected</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px;color:#4d7c0f;">{len(speakers)}</div>
      </div>
      <div style="flex:1;min-width:140px;background:#f9fafb;border:1px solid #e5e7eb;padding:12px 16px;border-radius:10px;">
        <div style="font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.04em;">Total Words</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px;color:#111827;">{len(result.words)}</div>
      </div>
      <div style="flex:1;min-width:140px;background:#f9fafb;border:1px solid #e5e7eb;padding:12px 16px;border-radius:10px;">
        <div style="font-size:12px;font-weight:600;color:#4b5563;text-transform:uppercase;letter-spacing:0.04em;">Inference Time</div>
        <div style="font-size:20px;font-weight:700;margin-top:2px;color:#111827;">{result.processing_time:.2f}s <span style="font-size:13px;font-weight:500;color:#4b5563;">(RTF {rtf:.2f}x)</span></div>
      </div>
    </div>
    """

    # 2. Multi-Track Speaker Diarization Swimlane Timeline
    swimlanes_html: list[str] = []
    for spk in speakers:
        accent, _, _ = _speaker_colors(spk)
        spk_segs = [seg for seg in result.diarization_segments if seg.speaker == spk]
        bars: list[str] = []
        for seg in spk_segs:
            left_pct = max(0.0, min(100.0, (seg.start / total_dur) * 100.0))
            width_pct = max(0.35, min(100.0 - left_pct, (seg.duration / total_dur) * 100.0))
            tooltip = html.escape(
                f"{spk}: {format_timestamp(seg.start)} - {format_timestamp(seg.end)} ({seg.duration:.2f}s)"
            )
            bars.append(
                f'<div title="{tooltip}" style="position:absolute;left:{left_pct:.2f}%;width:{width_pct:.2f}%;'
                f'top:3px;bottom:3px;background:{accent};border-radius:4px;opacity:0.9;"></div>'
            )
        swimlanes_html.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
              <div style="width:92px;font-size:13px;font-weight:600;color:#374151;flex-shrink:0;">{html.escape(spk)}</div>
              <div style="position:relative;flex:1;height:22px;background:#f3f4f6;border-radius:6px;overflow:hidden;border:1px solid #e5e7eb;">
                {''.join(bars)}
              </div>
            </div>
            """
        )

    timeline_section = f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin-bottom:18px;font-family:system-ui,-apple-system,sans-serif;">
      <div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:10px;">
        Speaker Activity Timeline (Nemotron-3-Diarization)
      </div>
      {''.join(swimlanes_html) if swimlanes_html else '<div style="color:#6b7280;font-size:13px;">No speaker activity intervals detected.</div>'}
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#6b7280;margin-top:6px;padding-left:102px;">
        <span>00:00.00</span>
        <span>{format_timestamp(total_dur * 0.25)}</span>
        <span>{format_timestamp(total_dur * 0.50)}</span>
        <span>{format_timestamp(total_dur * 0.75)}</span>
        <span>{format_timestamp(total_dur)}</span>
      </div>
    </div>
    """

    # 3. Color-Coded Speaker-Attributed Transcript Bubbles
    turn_cards: list[str] = []
    for turn in result.turns:
        accent, bg, text_col = _speaker_colors(turn.speaker)
        overlap_badge = (
            '<span style="font-size:11px;background:#fef2f2;color:#b91c1c;border:1px solid #fecaca;'
            'padding:1px 7px;border-radius:999px;margin-left:8px;">Overlap</span>'
            if turn.has_overlap
            else ""
        )
        word_spans: list[str] = []
        for w in turn.words:
            w_tip = html.escape(f"{w.start:.2f}s - {w.end:.2f}s ({w.speaker})")
            word_spans.append(f'<span title="{w_tip}">{html.escape(w.word)}</span>')
        body_html = " ".join(word_spans) if word_spans else html.escape(turn.text)

        turn_cards.append(
            f"""
            <div style="border-left:4px solid {accent};background:{bg};padding:12px 16px;border-radius:0 10px 10px 0;margin-bottom:10px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
                <div>
                  <span style="font-weight:700;font-size:13.5px;color:{text_col};">{html.escape(turn.speaker)}</span>
                  {overlap_badge}
                </div>
                <span style="font-family:monospace;font-size:12px;color:#4b5563;">
                  {format_timestamp(turn.start)} &rarr; {format_timestamp(turn.end)}
                </span>
              </div>
              <div style="font-size:15px;line-height:1.55;color:#111827;">{body_html}</div>
            </div>
            """
        )

    transcript_section = f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:16px;font-family:system-ui,-apple-system,sans-serif;">
      <div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:12px;">
        Speaker-Attributed Transcript (Canary-1B-v2 + Nemotron-3-Diarization)
      </div>
      {''.join(turn_cards) if turn_cards else '<div style="color:#6b7280;font-size:14px;">No speech transcribed.</div>'}
    </div>
    """

    return summary_html + timeline_section + transcript_section


def build_gradio_app(pipeline: DiarizationASRPipeline):
    """Create the interactive Gradio Blocks application."""
    import gradio as gr

    preset_choices = [(v["label"], k) for k, v in DIARIZATION_PRESETS.items()]
    default_language = os.environ.get("ASR_LANGUAGE", "de")
    default_batch_size = max(1, int(os.environ.get("ASR_BATCH_SIZE", "1")))
    default_max_chunk_sec = float(os.environ.get("ASR_MAX_CHUNK_SEC", "30.0"))

    def run_inference(
        audio_path: str | None,
        language: str,
        preset_key: str,
        batch_size: int,
        max_chunk_sec: float,
        merge_gap_threshold: float,
        nearest_tolerance: float,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ):
        if not audio_path:
            raise gr.Error("Please upload an audio file or record from your microphone first.")

        def update_progress(fraction: float, desc: str) -> None:
            progress(fraction, desc=desc)

        result = pipeline.process_audio(
            audio_path=audio_path,
            preset_key=preset_key,
            language=(language or default_language).strip(),
            nearest_tolerance=float(nearest_tolerance),
            merge_gap_threshold=float(merge_gap_threshold),
            max_chunk_sec=float(max_chunk_sec),
            batch_size=max(1, int(batch_size)),
            progress_callback=update_progress,
        )

        visual_html = render_visual_report(result)
        md_text = format_turns_as_markdown(result.turns)
        srt_text = format_turns_as_srt(result.turns)
        rttm_text = format_segments_as_rttm(
            result.diarization_segments, file_id=Path(audio_path).stem
        )
        json_text = result.to_json(indent=2)

        stats_rows = [
            [
                s.speaker,
                f"{s.total_duration:.2f}s",
                f"{s.percentage:.1f}%",
                s.turn_count,
                s.word_count,
            ]
            for s in result.speaker_stats
        ]

        # Write downloadable export files
        out_dir = Path(tempfile.mkdtemp(prefix="nemo_exports_"))
        stem = Path(audio_path).stem or "transcript"
        srt_file = out_dir / f"{stem}.srt"
        rttm_file = out_dir / f"{stem}.rttm"
        json_file = out_dir / f"{stem}.json"

        srt_file.write_text(srt_text, encoding="utf-8")
        rttm_file.write_text(rttm_text, encoding="utf-8")
        json_file.write_text(json_text, encoding="utf-8")

        return (
            visual_html,
            stats_rows,
            md_text,
            srt_text,
            rttm_text,
            json_text,
            [str(srt_file), str(rttm_file), str(json_file)],
        )

    with gr.Blocks(
        title="NVIDIA Nemotron-3-Diarization + Canary-1B-v2 Demo",
    ) as demo:
        gr.Markdown(
            f"""
            # Speaker-Attributed ASR Demo
            ### Powered by [`{DIARIZATION_MODEL_ID}`](https://huggingface.co/{DIARIZATION_MODEL_ID}) + [`{ASR_MODEL_ID}`](https://huggingface.co/{ASR_MODEL_ID})
            Upload an audio file or record directly from your microphone to run **end-to-end Sortformer speaker diarization** (up to 8 speakers with overlap detection) aligned with **language-locked Canary-1B-v2 word-level timestamps**.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Input Audio (File Upload or Microphone)",
                )
                language_dropdown = gr.Dropdown(
                    choices=SUPPORTED_ASR_LANGUAGES,
                    value=default_language,
                    allow_custom_value=True,
                    label="ASR Spoken Language (nvidia/canary-1b-v2)",
                    info="Explicitly locks source & target language (e.g. 'de' for German) so English loanwords are never translated into English.",
                )
                preset_dropdown = gr.Dropdown(
                    choices=preset_choices,
                    value="offline_high_accuracy",
                    label="Nemotron-3-Diarization Latency / Context Mode",
                    info="Offline mode uses a 30.4s window for maximum accuracy; streaming modes simulate low-latency chunked inference.",
                )
                with gr.Accordion("Chunking & Alignment Settings", open=False):
                    batch_size_slider = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=default_batch_size,
                        step=1,
                        label="ASR Batch Size (chunks per pass)",
                        info="Default 1 minimizes peak RAM on CPU and prevents padding waste; increase to 4–8 on GPU for faster throughput.",
                    )
                    chunk_sec_slider = gr.Slider(
                        minimum=10.0,
                        maximum=120.0,
                        value=default_max_chunk_sec,
                        step=5.0,
                        label="ASR Max Chunk Duration (seconds)",
                        info="Splits long audio at silence gaps between diarization segments for faster CPU inference and per-chunk progress.",
                    )
                    merge_gap_slider = gr.Slider(
                        minimum=0.5,
                        maximum=10.0,
                        value=4.0,
                        step=0.5,
                        label="Same-Speaker Turn Merge Threshold (seconds)",
                        info="Maximum silence gap between words of the same speaker before splitting into a new turn.",
                    )
                    tolerance_slider = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="Nearest Speaker Fallback Window (seconds)",
                        info="Maximum distance to match words falling just outside diarization segment boundaries.",
                    )

                run_btn = gr.Button(
                    "Transcribe & Diarize Audio",
                    variant="primary",
                    size="lg",
                )

                stats_table = gr.Dataframe(
                    headers=["Speaker", "Speaking Time", "Share %", "Turns", "Words"],
                    datatype=["str", "str", "str", "number", "number"],
                    label="Speaker Analytics",
                    interactive=False,
                )
                download_files = gr.File(
                    label="Download Exports (.srt, .rttm, .json)",
                    interactive=False,
                )

            with gr.Column(scale=7):
                with gr.Tabs():
                    with gr.TabItem("Interactive Timeline & Transcript"):
                        visual_output = gr.HTML(
                            value="<div style='padding:24px;color:#6b7280;text-align:center;border:1px dashed #d1d5db;border-radius:12px;'>"
                            "Upload or record audio and click <b>Transcribe & Diarize Audio</b> to view the speaker timeline and attributed transcript."
                            "</div>"
                        )
                    with gr.TabItem("Markdown"):
                        md_output = gr.Code(label="Markdown Transcript", language="markdown")
                    with gr.TabItem("SRT Subtitles"):
                        srt_output = gr.Code(label="SRT Subtitles", language=None)
                    with gr.TabItem("RTTM (Diarization)"):
                        rttm_output = gr.Code(label="NIST RTTM Output", language=None)
                    with gr.TabItem("Full JSON"):
                        json_output = gr.Code(label="Aligned Word & Turn JSON", language="json")

        run_btn.click(
            fn=run_inference,
            inputs=[
                audio_input,
                language_dropdown,
                preset_dropdown,
                batch_size_slider,
                chunk_sec_slider,
                merge_gap_slider,
                tolerance_slider,
            ],
            outputs=[
                visual_output,
                stats_table,
                md_output,
                srt_output,
                rttm_output,
                json_output,
                download_files,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVIDIA Nemotron-3-Diarization + Canary-1B-v2 Demo"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Optional path to an audio file for CLI batch mode (if omitted, launches the Gradio Web UI).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.environ.get("ASR_LANGUAGE", "de"),
        help="Spoken language ISO code for nvidia/canary-1b-v2 (default: ASR_LANGUAGE env or 'de').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("ASR_BATCH_SIZE", "1")),
        help="ASR batch size for Canary chunk transcription (default: ASR_BATCH_SIZE env or 1).",
    )
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=float(os.environ.get("ASR_MAX_CHUNK_SEC", "30.0")),
        help="Maximum ASR chunk duration in seconds (default: ASR_MAX_CHUNK_SEC env or 30.0).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="offline_high_accuracy",
        choices=list(DIARIZATION_PRESETS.keys()),
        help="Sortformer diarization latency preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write .md, .srt, .rttm, and .json outputs when running in CLI mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. 'cuda', 'cpu'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        help="Gradio server host address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        help="Gradio server port (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload NeMo models at startup instead of on first request.",
    )
    args = parser.parse_args()

    # Propagate CLI flags to environment so build_gradio_app picks them up as UI defaults
    os.environ["ASR_LANGUAGE"] = args.language
    os.environ["ASR_BATCH_SIZE"] = str(max(1, args.batch_size))
    os.environ["ASR_MAX_CHUNK_SEC"] = str(args.max_chunk_sec)

    pipeline = DiarizationASRPipeline(device=args.device)

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = pipeline.process_audio(
            audio_path=audio_path,
            preset_key=args.preset,
            language=args.language,
            max_chunk_sec=args.max_chunk_sec,
            batch_size=max(1, args.batch_size),
        )
        md_text = format_turns_as_markdown(result.turns)
        print("\n=== Speaker-Attributed Transcript ===\n")
        print(md_text)

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = audio_path.stem
            (out_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")
            (out_dir / f"{stem}.srt").write_text(
                format_turns_as_srt(result.turns), encoding="utf-8"
            )
            (out_dir / f"{stem}.rttm").write_text(
                format_segments_as_rttm(result.diarization_segments, file_id=stem),
                encoding="utf-8",
            )
            (out_dir / f"{stem}.json").write_text(result.to_json(indent=2), encoding="utf-8")
            logger.info("Saved outputs to %s", out_dir)
        return

    if args.preload:
        pipeline.load_models()

    demo = build_gradio_app(pipeline)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
