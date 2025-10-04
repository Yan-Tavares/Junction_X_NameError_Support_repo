import argparse, json, subprocess, re
from pathlib import Path
import numpy as np
import torch, torchaudio
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, AutoModel
from scipy.signal import find_peaks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------
# Helpers: download + ASR
# --------------------------
def download_audio(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    templ = out_dir / "%(id)s.%(ext)s"
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0", "-o", str(templ), url]
    subprocess.run(cmd, check=True)
    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError("No WAV produced by yt-dlp.")
    return wavs[0]


def whisper_transcribe(path: Path, model_size="medium"):
    import librosa
    import numpy as np
    import soundfile as sf
    from scipy import signal
    
    # Load and preprocess audio
    audio, sr = librosa.load(str(path), sr=16000)
    
    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Normalize audio
    emphasized_audio = librosa.util.normalize(emphasized_audio)
    
    # Apply noise reduction using spectral gating
    S = librosa.stft(emphasized_audio)
    mag = np.abs(S)
    
    # Estimate noise floor
    noise_floor = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1]*0.1)], axis=1)
    noise_floor = noise_floor[:, np.newaxis]
    
    # Apply soft thresholding
    gain = (mag - noise_floor) / mag
    gain = np.maximum(0, gain)
    
    # Apply gain reduction
    S_clean = S * gain
    
    # Inverse STFT
    audio_clean = librosa.istft(S_clean)
    
    # Save preprocessed audio
    temp_path = path.parent / (path.stem + "_processed.wav")
    sf.write(str(temp_path), audio_clean, sr)
    
    model = WhisperModel(
        model_size,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        compute_type=("float16" if torch.cuda.is_available() else "int8"),
        download_root=None
    )
    # First pass: Get full transcription with more aggressive VAD
    segments, _ = model.transcribe(
        str(temp_path),
        word_timestamps=True,  # Get word timestamps
        condition_on_previous_text=True,  # Help with context
        language="en",
        task="transcribe",
        beam_size=10,  # Increase beam size for better accuracy
        best_of=10,
        temperature=0.0,  # Be deterministic
        initial_prompt="This is a conversation or speech that may contain emotional content.",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=100,  # Very short silence duration
            speech_pad_ms=30,  # Minimal padding
            threshold=0.2  # More sensitive to speech
        )
    )
    
    # Second pass with different VAD settings
    segments_with_times, _ = model.transcribe(
        str(temp_path),
        word_timestamps=True,
        condition_on_previous_text=True,
        language="en",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=100,
            threshold=0.3
        )
    )
    
    # Combine results to get more accurate segments
    combined_segments = []
    for seg1, seg2 in zip(segments, segments_with_times):
        # Use timing from first pass (more aggressive VAD) but verify with second pass
        if abs(seg1.start - seg2.start) < 0.5:  # If segments roughly align
            combined_segments.append({
                "start": float(seg1.start),
                "end": float(seg1.end),
                "text": seg1.text.strip()
            })
    
    # Clean up temporary file
    temp_path.unlink()
    
    # If no segments were aligned, fall back to first pass results
    if not combined_segments:
        combined_segments = [{
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        } for seg in segments]
    
    # Get the full text by combining all segments
    full_text = " ".join(seg["text"] for seg in combined_segments)
    
    return full_text, combined_segments
    
    # Combine results: Use timing from second pass but text from first pass
    # for better accuracy
    aligned_segments = []
    first_pass_texts = [seg.text.strip() for seg in segments]
    full_text = " ".join(first_pass_texts)
    
    for seg in segments_with_times:
        aligned_segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
    
    return full_text, aligned_segments
    sents = [{"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()} for seg in segments]
    full_text = " ".join(s["text"] for s in sents).strip()
    return full_text, sents


# --------------------------
# Helpers: text classification
# --------------------------
def load_text_model(path_or_hub: str):
    tok = AutoTokenizer.from_pretrained(path_or_hub)
    clf = AutoModelForSequenceClassification.from_pretrained(path_or_hub).to(DEVICE).eval()
    return tok, clf


def classify_sentences(sents, tok, clf):
    texts = [s["text"] for s in sents]
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(clf(**batch).logits, dim=-1).cpu().numpy()
    return probs


# --------------------------
# Helpers: audio emotion
# --------------------------
def load_audio(path: Path, sr=16000):
    wav, s = torchaudio.load(str(path))
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
        s = sr
    wav = wav.mean(0, keepdim=True)
    return wav, s


def load_emo_model():
    proc = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    mdl = AutoModel.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim").to(DEVICE).eval()
    return proc, mdl


def emo_timeseries(wav, sr, proc, mdl, win_s=2.0, hop_s=0.5):
    wav = wav.squeeze(0)
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n = max(0, (len(wav) - win) // hop + 1)
    times, vals = [], []
    for i in range(n):
        s = i * hop
        e = s + win
        chunk = wav[s:e]
        inputs = proc(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs).logits.squeeze(0).cpu().numpy()  # [arousal, dominance, valence]
        times.append(((s + e) / 2) / sr)
        vals.append({"arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])})
    return times, vals


def find_arousal_peaks(times, vals, height=0.7, min_sep_s=2.0, hop_s=0.5):
    series = np.array([v["arousal"] for v in vals]) if vals else np.array([])
    if len(series) == 0:
        return []
    distance = max(1, int(min_sep_s / hop_s))
    idx, _ = find_peaks(series, height=height, distance=distance)
    return [{"t": float(times[i]), "arousal": float(series[i])} for i in idx]


def which_sentence(spans, t):
    for s in spans:
        if s["start"] <= t <= s["end"]:
            return s
    return None


# --------------------------
# New: text normalization + merge short segments
# --------------------------
def _norm_text(t: str) -> str:
    # Remove excessive letter repetitions
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    # Remove repeated syllables
    t = re.sub(r"([a-z]{1,3})\1{2,}", r"\1", t, flags=re.IGNORECASE)
    # Clean up spaces
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def merge_short_segments(segments, min_dur=0.5, min_tokens=2, max_gap=0.2):
    """Łączy za krótkie segmenty w większe fragmenty, by model dostawał sensowne zdania."""
    if not segments:
        return segments

    # normalizuj tekst
    for s in segments:
        s["text"] = _norm_text(s["text"])

    merged = []
    cur = dict(segments[0])
    for nxt in segments[1:]:
        gap = max(0.0, float(nxt["start"]) - float(cur["end"]))
        cur_dur = float(cur["end"]) - float(cur["start"])
        cur_tokens = len(cur["text"].split())
        need_merge = (cur_dur < min_dur) or (cur_tokens < min_tokens) or (gap <= max_gap)
        if need_merge:
            cur["end"] = float(nxt["end"])
            cur["text"] = (cur["text"] + " " + nxt["text"]).strip()
        else:
            merged.append(cur)
            cur = dict(nxt)
    merged.append(cur)

    # drugi przebieg na wszelki wypadek
    out = []
    cur = merged[0]
    for nxt in merged[1:]:
        cur_dur = float(cur["end"]) - float(cur["start"])
        cur_tokens = len(cur["text"].split())
        if cur_dur < min_dur or cur_tokens < min_tokens:
            cur["end"] = float(nxt["end"])
            cur["text"] = (cur["text"] + " " + nxt["text"]).strip()
        else:
            out.append(cur)
            cur = nxt
    out.append(cur)
    return out


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--url", type=str)
    grp.add_argument("--path", type=str)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--text_model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--whisper_size", default="medium")
    ap.add_argument("--arousal_peak", type=float, default=0.7)
    ap.add_argument("--fast_debug", action="store_true", help="Smaller models / fewer computations")
    ap.add_argument("--skip_emo", action="store_true", help="Skip audio emotion model")
    args = ap.parse_args()

    # default example if nothing passed
    if not args.url and not args.path:
        example_url = "https://www.youtube.com/watch?v=hkSj-QapfZo"
        print(f"No --url/--path provided — using example video: {example_url}")
        args.url = example_url

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    audio_path = download_audio(args.url, out_dir) if args.url else Path(args.path)

    # fast debug overrides (OPTIONAL)
    if args.fast_debug:
        args.whisper_size = "small"      # smaller ASR
        # only override text model in fast debug if user left default
        if args.text_model == "cardiffnlp/twitter-roberta-base-hate-latest":
            args.text_model = "distilbert-base-uncased-finetuned-sst-2-english"
        print(f"FAST DEBUG: whisper={args.whisper_size}, text_model={args.text_model}")

    # 1) ASR
    _, sents = whisper_transcribe(audio_path, model_size=args.whisper_size)
    # Filter out segments with mostly repeated characters
    sents = [s for s in sents if len(set(s["text"].lower())) >= 3]
    sents = merge_short_segments(sents, min_dur=0.3, min_tokens=1, max_gap=0.1)

    # 2) Text classifier
    tok, clf = load_text_model(args.text_model)
    probs = classify_sentences(sents, tok, clf)
    
    # Debug: print probabilities
    print("\nDebug - Classification probabilities:")
    for i, (s, p) in enumerate(zip(sents, probs)):
        print(f"\nSegment {i+1}:")
        print(f"Text: {s['text']}")
        print(f"Probabilities: {p}")
        print(f"Max probability: {p.max():.4f} for class {p.argmax()}")


    labels = ["normal", "offensive", "hatespeech"]  # Odpowiednie klasy z modelu
    hate_spans = []
    for s, p in zip(sents, probs):
        p = p.astype(float)
        li = int(p.argmax())
        conf_max = float(p[li])

        # Pobierz prawdopodobieństwa dla każdej klasy
        p_normal = float(p[0])
        p_offensive = float(p[1])
        p_hatespeech = float(p[2])
        
        # Ustal pewność i etykietę
        if p_hatespeech >= 0.45:
            label = "hatespeech"
            conf = p_hatespeech
        elif p_offensive >= 0.45:
            label = "offensive"
            conf = p_offensive
        elif p_normal >= 0.45:
            label = "normal"
            conf = p_normal
        else:
            # Jeśli żadna klasa nie przekracza progu, weź najwyższą
            max_prob = max(p_normal, p_offensive, p_hatespeech)
            if max_prob == p_hatespeech:
                label = "hatespeech"
                conf = p_hatespeech
            elif max_prob == p_offensive:
                label = "offensive"
                conf = p_offensive
            else:
                label = "normal"
                conf = p_normal

        hate_spans.append({**s, "label": label, "confidence": float(conf)})

    # 3) Emotions (optional)
    wav, sr = load_audio(audio_path, 16000)
    times, vals = [], []
    if not args.skip_emo:
        if args.fast_debug:
            emo_win_s, emo_hop_s = 4.0, 2.0
        else:
            emo_win_s, emo_hop_s = 2.0, 0.5
        emo_proc, emo_mdl = load_emo_model()
        times, vals = emo_timeseries(wav, sr, emo_proc, emo_mdl, win_s=emo_win_s, hop_s=emo_hop_s)
        hop_for_peaks = emo_hop_s
    else:
        print("Skipping emotion model inference (--skip_emo).")
        hop_for_peaks = 0.5

    peaks = find_arousal_peaks(times, vals, height=args.arousal_peak, min_sep_s=2.0, hop_s=hop_for_peaks)

    emo_events = []
    for pk in peaks:
        s = which_sentence(hate_spans, pk["t"])
        emo_events.append({
            "t": pk["t"],
            "arousal": pk["arousal"],
            "coincides_with": (s["label"] if s else None),
            "text": (s["text"] if s else None),
            "span_start": (s["start"] if s else None),
            "span_end": (s["end"] if s else None),
        })

    # 4) Save
    result = {
        "audio_path": str(audio_path),
        "hate_spans": hate_spans,
        "emotion_times": [{"t": float(t), **v} for t, v in zip(times, vals)],
        "emotion_peaks": emo_events,
    }
    out_json = out_dir / "analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
