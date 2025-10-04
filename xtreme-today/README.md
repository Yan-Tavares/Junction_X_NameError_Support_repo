# Extreme Speech & Emotion – *Today-ready* Toolkit

Train a text hate/offense classifier now (no special access) and analyze audio/video end-to-end:
- Download audio from URL (YouTube etc., respect TOS/licensing)
- Transcribe with Whisper (timestamps)
- Classify hate/offense per segment
- Run speech emotion (valence/arousal/dominance) over time
- Emit timestamps for peaks and align them to segments

## Files
- `train_text_hatexplain.py` – fine-tunes `roberta-base` on HateXplain (3 classes: hate/offensive/normal).
- `analyze_media_full.py` – URL or file -> ASR (Faster-Whisper) -> hate spans + emotion time-series/peaks.
- `requirements.txt` – pip deps.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Analyze a URL (audio will be extracted as .wav)
python analyze_media_full.py --url "https://www.youtube.com/watch?v=XXXXXXXX"

# Or a local media file
python analyze_media_full.py --path /path/to/file.mp4
```

## Train your text model (HateXplain)
```bash
source .venv/bin/activate
python train_text_hatexplain.py --epochs 2 --batch 16 --out models/hatexplain-roberta-mini
# then use it:
python analyze_media_full.py --path sample.wav --text_model models/hatexplain-roberta-mini
```

Notes:
- Use HateMM (Zenodo) to evaluate time-localized hate on AV content.
- For word-level timing use WhisperX instead of Faster-Whisper.
- Emotion model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (research use).
