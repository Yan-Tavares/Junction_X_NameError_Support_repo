import argparse, json, subprocess
from pathlib import Path
import numpy as np
import torch, torchaudio
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, AutoModel
from scipy.signal import find_peaks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_audio(url:str, out_dir:Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    templ = out_dir / "%(id)s.%(ext)s"
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0", "-o", str(templ), url]
    subprocess.run(cmd, check=True)
    wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs: raise RuntimeError("No WAV produced by yt-dlp.")
    return wavs[0]

def whisper_transcribe(path:Path, model_size="medium"):
    model = WhisperModel(model_size, device=("cuda" if torch.cuda.is_available() else "cpu"),
                         compute_type=("float16" if torch.cuda.is_available() else "int8"))
    segments, _ = model.transcribe(str(path), word_timestamps=False, vad_filter=True)
    sents = [{"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()} for seg in segments]
    full_text = " ".join(s['text'] for s in sents).strip()
    return full_text, sents

def load_text_model(path_or_hub:str):
    tok = AutoTokenizer.from_pretrained(path_or_hub)
    clf = AutoModelForSequenceClassification.from_pretrained(path_or_hub).to(DEVICE).eval()
    return tok, clf

def classify_sentences(sents, tok, clf):
    texts = [s["text"] for s in sents]
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(clf(**batch).logits, dim=-1).cpu().numpy()
    return probs

def load_audio(path:Path, sr=16000):
    wav, s = torchaudio.load(str(path))
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
        s = sr
    wav = wav.mean(0, keepdim=True)
    return wav, s

def load_emo_model():
    proc = AutoProcessor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    mdl  = AutoModel.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim").to(DEVICE).eval()
    return proc, mdl

def emo_timeseries(wav, sr, proc, mdl, win_s=2.0, hop_s=0.5):
    wav = wav.squeeze(0)
    win = int(win_s*sr); hop = int(hop_s*sr)
    n = max(0, (len(wav)-win)//hop + 1)
    times, vals = [], []
    for i in range(n):
        s = i*hop; e = s+win
        chunk = wav[s:e]
        inputs = proc(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k,v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs).logits.squeeze(0).cpu().numpy()  # [arousal, dominance, valence]
        times.append(((s+e)/2)/sr)
        vals.append({"arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])})
    return times, vals

def find_arousal_peaks(times, vals, height=0.7, min_sep_s=2.0, hop_s=0.5):
    series = np.array([v["arousal"] for v in vals]) if vals else np.array([])
    if len(series)==0: return []
    import math
    distance = max(1, int(min_sep_s / hop_s))
    idx, _ = find_peaks(series, height=height, distance=distance)
    return [{"t": float(times[i]), "arousal": float(series[i])} for i in idx]

def which_sentence(spans, t):
    for s in spans:
        if s["start"] <= t <= s["end"]:
            return s
    return None

def main():
    ap = argparse.ArgumentParser()
    # let the script run with an example video when no args are given
    grp = ap.add_mutually_exclusive_group(required=False)
    grp.add_argument("--url", type=str)
    grp.add_argument("--path", type=str)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--text_model", default="cardiffnlp/twitter-roberta-base-hate-latest")
    ap.add_argument("--whisper_size", default="medium")
    ap.add_argument("--arousal_peak", type=float, default=0.7)
    # Fast debug mode: switch to smaller/faster models & reduced processing so you can iterate quickly.
    # Changes made when --fast_debug is set (comments list original -> new):
    # - Whisper model: "medium" (~1.8GB) -> "small" (~244MB) or "tiny" (~39MB) for fastest runs
    # - Text model: "cardiffnlp/twitter-roberta-base-hate-latest" (large, ~500MB) ->
    #       "distilbert-base-uncased-finetuned-sst-2-english" (DistilBERT, ~250MB) for faster load
    # - Emotion model: "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" (large) -> skipped or
    #       use much larger hop/window to reduce inference calls
    ap.add_argument("--fast_debug", action="store_true", help="Use smaller models and fewer computations for fast debugging")
    ap.add_argument("--skip_emo", action="store_true", help="Skip loading/running the audio emotion model (fastest)")
    args = ap.parse_args()

    # If the user didn't provide a URL or path, fall back to an example YouTube video
    if not args.url and not args.path:
        example_url = "https://www.youtube.com/watch?v=7QkJ6IYikWo"
        print(f"No --url/--path provided — using example video: {example_url}")
        args.url = example_url

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    audio_path = download_audio(args.url, out_dir) if args.url else Path(args.path)

    # If fast_debug requested, override some heavy defaults for quicker iteration
    if args.fast_debug:
        # Whisper: medium -> small (or tiny for fastest). small is a very reasonable balance.
        # Original default: "medium" (much larger). New default used here: "small" (~244MB)
        args.whisper_size = "small"
        # Text model: heavy hate-detector -> DistilBERT SST2 (fast to load and run)
        # Original default: "cardiffnlp/twitter-roberta-base-hate-latest" -> switched to:
        args.text_model = "distilbert-base-uncased-finetuned-sst-2-english"
        print("FAST DEBUG: using smaller models (whisper=small, text_model=distilbert-sst2)")

    _, sents = whisper_transcribe(audio_path, model_size=args.whisper_size)

    tok, clf = load_text_model(args.text_model)
    probs = classify_sentences(sents, tok, clf)
    labels = ["non-hate", "hate"] if probs.shape[1]==2 else [f"class_{i}" for i in range(probs.shape[1])]
    hate_spans = []
    for s, p in zip(sents, probs):
        li = int(p.argmax()); conf = float(p[li])
        hate_spans.append({**s, "label": labels[li], "confidence": conf})

    wav, sr = load_audio(audio_path, 16000)
    times, vals = [], []
    if not args.skip_emo:
        if args.fast_debug:
            # In fast_debug reduce number of windows by increasing window & hop sizes
            # Original: win_s=2.0, hop_s=0.5 -> many windows
            # Fast: win_s=4.0, hop_s=2.0 -> ~1/4 the number of windows -> much faster
            emo_win_s, emo_hop_s = 4.0, 2.0
        else:
            emo_win_s, emo_hop_s = 2.0, 0.5
        emo_proc, emo_mdl = load_emo_model()
        times, vals = emo_timeseries(wav, sr, emo_proc, emo_mdl, win_s=emo_win_s, hop_s=emo_hop_s)
    else:
        print("Skipping emotion model inference (--skip_emo).")
    # Use hop_s consistent with how we produced the times/vals
    hop_for_peaks = 2.0 if (args.fast_debug and not args.skip_emo) else 0.5
    peaks = find_arousal_peaks(times, vals, height=args.arousal_peak, min_sep_s=2.0, hop_s=hop_for_peaks)

    emo_events = []
    for pk in peaks:
        s = which_sentence(hate_spans, pk["t"])
        emo_events.append({
            "t": pk["t"], "arousal": pk["arousal"],
            "coincides_with": (s["label"] if s else None),
            "text": (s["text"] if s else None),
            "span_start": (s["start"] if s else None),
            "span_end": (s["end"] if s else None),
        })

    result = {
        "audio_path": str(audio_path),
        "hate_spans": hate_spans,
        "emotion_times": [{"t": float(t), **v} for t, v in zip(times, vals)],
        "emotion_peaks": emo_events
    }
    out_json = out_dir / "analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_json}")

if __name__ == "__main__":
    main()
