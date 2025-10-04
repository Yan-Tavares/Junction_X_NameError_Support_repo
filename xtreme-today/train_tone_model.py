import os
import torch
import numpy as np
import librosa
from datasets import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate

# -------------------------
# 1. Device setup
# -------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# -------------------------
# 2. Load dataset (RAVDESS raw structure)
# -------------------------
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

src_root = "/Users/inesmarques/Desktop/Junction_X_NameError_Support_repo/xtreme-today/ravdess_raw"  # change if needed
data = []

for actor in os.listdir(src_root):
    actor_path = os.path.join(src_root, actor)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) > 2:
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    data.append({
                        "file": os.path.join(actor_path, file),
                        "label": emotion_map[emotion_code]
                    })

print(f"✅ Loaded {len(data)} files with labels: {set([d['label'] for d in data])}")

# -------------------------
# 3. Label mappings
# -------------------------
labels = sorted(list(set([d["label"] for d in data])))
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

# Map labels to integers
for d in data:
    d["label"] = label2id[d["label"]]

dataset = Dataset.from_list(data)

# -------------------------
# 4. Load pretrained model
# -------------------------
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

# Freeze encoder for faster training
for param in model.wav2vec2.parameters():
    param.requires_grad = False

# -------------------------
# 5. Preprocess function (librosa loader)
# -------------------------
def preprocess(batch):
    audio, _ = librosa.load(batch["file"], sr=16000, mono=True)  # always mono
    audio = np.squeeze(audio)  # force 1D

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        max_length=16000 * 5,  # clip/pad to 5 seconds
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return {
        "input_values": inputs["input_values"].squeeze(0),  # shape [seq_len]
        "labels": batch["label"]
    }



dataset = dataset.map(preprocess, remove_columns=["file"])
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -------------------------
# 6. Training setup
# -------------------------
training_args = TrainingArguments(
    output_dir="./emotion_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=5,
    load_best_model_at_end=True,     # needed
    metric_for_best_model="f1",      # tell Trainer which metric to monitor
    greater_is_better=True,          # because higher f1 = better
)



# -------------------------
# 7. Metrics
# -------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_score = f1.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# -------------------------
# 8. Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------------
# 9. Train
# -------------------------
trainer.train()

# -------------------------
# 10. Save model
# -------------------------
trainer.save_model("./fine_tuned_emotion_model")
print("🎉 Fine-tuned model saved at ./fine_tuned_emotion_model")
