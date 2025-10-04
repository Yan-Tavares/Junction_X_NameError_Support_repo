import gradio as gr
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

whisper = WhisperModel("medium")   # pick size per your GPU/CPU
tok = AutoTokenizer.from_pretrained("your-hf-org/xtremespeech-3way-roberta")
clf = AutoModelForSequenceClassification.from_pretrained("your-hf-org/xtremespeech-3way-roberta").eval()

def analyze(audio):
    segments, _ = whisper.transcribe(audio, word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"text": w.word, "start": w.start, "end": w.end})
    transcript = " ".join(w["text"] for w in words)

    inputs = tok(transcript, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probs = torch.softmax(clf(**inputs).logits, dim=-1)[0].tolist()
    labels = ["derogatory","exclusionary","dangerous"]
    pred = labels[int(probs.index(max(probs)))]

    # simple span demo: highlight exact toxic word matches (replace with attributions later)
    toxic = {"derogatory":{"idiot","stupid"}, "exclusionary":{"ban","keep"}, "dangerous":{"kill","attack"}}
    spans = [w for w in words if w["text"].lower().strip(".,?!") in toxic.get(pred, set())]

    return transcript, {"label": pred, "probs": dict(zip(labels, probs)), "spans": spans}

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(sources=["microphone","upload"], type="filepath"),
    outputs=[gr.Textbox(label="Transcript"),
             gr.Json(label="Prediction + timestamped spans")],
    title="Vocal Firewall – Demo"
)
if __name__ == "__main__":
    demo.launch()
