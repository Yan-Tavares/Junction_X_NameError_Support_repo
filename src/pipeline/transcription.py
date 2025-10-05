"""
Audio transcription module for Vocal Firewall.
Handles Whisper-based speech-to-text with smart segmentation.
"""

import re
from pathlib import Path
from faster_whisper import WhisperModel
import torch


class AudioTranscriber:
    """Handles audio transcription using Whisper with smart segmentation."""
    
    def __init__(self, model_size="medium", device=None):
        """
        Initialize the transcriber with Whisper model.
        
        Args:
            model_size: Whisper model size (tiny/base/small/medium/large)
            device: Device to run on (cuda/cpu, None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        self.transcriber = WhisperModel(
            model_size,
            device=self.device,
            compute_type=compute_type,
            download_root=None
        )
    
    def transcribe(self, audio_path, language="en", beam_size=5):
        """
        Transcribe audio using Whisper with smart segmentation.
        
        Args:
            audio_path: Path to audio file (preprocessed)
            language: Language code (default: "en")
            beam_size: Beam size for decoding (default: 5)
            
        Returns:
            tuple: (full_transcript, list of segment dicts with start/end/text)
        """
        print("🎤 Transcribing audio...")
        
        # Transcribe with Whisper
        segments, _ = self.transcriber.transcribe(
            str(audio_path),
            word_timestamps=True,
            condition_on_previous_text=True,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=100,
                threshold=0.3
            )
        )
        
        # Convert to list of dicts
        segment_list = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            }
            for seg in segments
        ]
        
        # Filter out segments with mostly repeated characters
        segment_list = [
            s for s in segment_list 
            if len(set(s["text"].lower())) >= 3 and s["text"].strip()
        ]
        
        # Merge short segments for better classification
        segment_list = merge_short_segments(segment_list)
        
        # Clean up preprocessed file if it's temporary
        audio_path = Path(audio_path)
        if audio_path.exists() and "_processed" in str(audio_path):
            audio_path.unlink()
        
        # Build full transcript
        full_transcript = " ".join(s["text"] for s in segment_list)
        
        print(f"✅ Transcribed {len(segment_list)} segments")
        return full_transcript, segment_list


def merge_short_segments(segments, min_dur=0.3, min_tokens=1, max_gap=0.1, max_text_len=350):
    """
    Merge short segments to create more meaningful chunks for classification.
    
    Args:
        segments: List of segment dicts with start, end, text
        min_dur: Minimum duration in seconds
        min_tokens: Minimum number of tokens
        max_gap: Maximum gap between segments to merge (seconds)
        max_text_len: Maximum text length to avoid overly long segments
        
    Returns:
        List of merged segments
    """
    if not segments:
        return segments
    
    # Normalize text
    for seg in segments:
        seg["text"] = normalize_text(seg["text"])
    
    merged = []
    cur = dict(segments[0])
    
    for nxt in segments[1:]:
        gap = max(0.0, float(nxt["start"]) - float(cur["end"]))
        new_text = (cur["text"] + " " + nxt["text"]).strip()
        cur_dur = float(cur["end"]) - float(cur["start"])
        cur_tokens = len(cur["text"].split())
        
        # Decide whether to merge
        too_short = cur_dur < min_dur or cur_tokens < min_tokens
        small_gap = gap <= max_gap
        within_length = len(new_text) < max_text_len
        
        if (too_short or small_gap) and within_length:
            cur["end"] = float(nxt["end"])
            cur["text"] = new_text
        else:
            merged.append(cur)
            cur = dict(nxt)
    
    merged.append(cur)
    return merged


def normalize_text(text):
    """
    Normalize text by removing excessive repetitions.
    
    Args:
        text: Input text string
        
    Returns:
        str: Normalized text
    """
    # Remove excessive character repetitions (e.g., "hellooo" -> "hello")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    # Remove repeated syllables (e.g., "lalala" -> "la")
    text = re.sub(r"([a-z]{1,3})\1{2,}", r"\1", text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_timestamps(segments):
    """
    Extract timestamps from segments for model processing.
    
    Args:
        segments: List of segment dicts with start, end, text
        
    Returns:
        list: List of timestamps (start of each segment + end of last segment)
    """
    if not segments:
        return []
    return [seg["start"] for seg in segments] + [segments[-1]["end"]]
