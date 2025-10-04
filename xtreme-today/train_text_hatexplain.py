import argparse
import random
import numpy as np
import torch

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
from sklearn.metrics import f1_score, classification_report
import torch.nn as nn
import inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="models/hatexplain-roberta-ft")
    ap.add_argument("--fast_debug", action="store_true",
                    help="Use smaller model / fewer epochs for fast iteration")
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="Limit number of training samples (debug only)")
    args = ap.parse_args()

    # reproducibility
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1) Load data (fallback to synthetic if local FS not supported)
    try:
        ds = load_dataset("Hate-speech-CNERG/hatexplain")
    except NotImplementedError:
        print("Warning: LocalFileSystem cache unsupported; using small synthetic dataset for debug.")
        from datasets import Dataset, DatasetDict

        def make_synth(n):
            items = []
            for i in range(n):
                items.append({"post_tokens": ["this", "is", "sample", f"{i}"], "annotators": {"label": [i % 3]}})
            return Dataset.from_list(items)

        ds = DatasetDict({"train": make_synth(2000), "validation": make_synth(500), "test": make_synth(500)})

    # 2) Map to (text, label)
    def to_text(example):
        text = " ".join(example["post_tokens"])
        label = example["annotators"]["label"][0] if isinstance(example["annotators"]["label"], list) else example["annotators"]["label"]
        return {"text": text, "label": int(label)}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    # 3) Optional subsetting for quick debug
    if args.max_train_samples is not None:
        print(f"Subsetting train set to first {args.max_train_samples} samples (debug).")
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(200, len(ds["validation"]))))
        ds["test"] = ds["test"].select(range(min(200, len(ds["test"]))))

    # 4) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    def tok_fn(batch): return tok(batch["text"], truncation=True, padding="max_length", max_length=128)
    ds_tok = ds.map(tok_fn, batched=True)

    # 5) Metrics
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        macro_f1 = f1_score(p.label_ids, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    # 6) Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(DEVICE)

    # 7) Class weights (imbalance-aware)
    num_labels = 3
    train_labels = np.array(ds["train"]["label"], dtype=int)
    counts = np.bincount(train_labels, minlength=num_labels).astype(float)
    class_weights = (counts.sum() / (counts + 1e-9))
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float)
    print("Class counts:", counts.tolist(), " -> weights:", class_weights.tolist())

    # 8) TrainingArguments (filter keys to avoid version mismatches)
    fp16_flag = bool(torch.cuda.is_available())  # use fp16 only on CUDA
    # Set some debug defaults
    if args.fast_debug:
        n_epochs = 1
        eval_strategy = "no"
        use_early_stopping = False
    else:
        n_epochs = args.epochs
        eval_strategy = "epoch"
        use_early_stopping = True

    desired_args = {
        "output_dir": args.out,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": args.batch,
        "num_train_epochs": n_epochs,
        "weight_decay": 0.01,
        "evaluation_strategy": eval_strategy,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "fp16": fp16_flag,
        "dataloader_num_workers": 4,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    accepted = set(sig.parameters.keys()); accepted.discard("self")
    filtered = {k: v for k, v in desired_args.items() if k in accepted}
    dropped = [k for k in desired_args if k not in filtered]
    if dropped:
        print(f"TrainingArguments: dropped unsupported args: {dropped}")
    if "evaluation_strategy" not in filtered and filtered.get("load_best_model_at_end", False):
        print("Evaluation strategy unsupported; disabling load_best_model_at_end.")
        filtered["load_best_model_at_end"] = False
    args_tr = TrainingArguments(**filtered)

    # 9) Weighted trainer with early stopping
    class WeightedTrainer(Trainer):
        def __init__(self, *targs, class_weights=None, **kwargs):
            super().__init__(*targs, **kwargs)
            self.class_weights = class_weights
            self.num_labels = getattr(self.model.config, "num_labels", num_labels)
            self.ce = nn.CrossEntropyLoss(weight=self.class_weights).to(self.model.device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights_t.to(model.device),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if use_early_stopping else [],
        # Keep using tokenizer for now as processing_class is not fully supported yet
        tokenizer=tok,
    )

    # 10) Train + save
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    # 11) Evaluate on test
    preds = trainer.predict(ds_tok["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    label_names = ["hatespeech", "offensive", "normal"]
    print(classification_report(y_true, y_pred, target_names=label_names))


if __name__ == "__main__":
    main()
