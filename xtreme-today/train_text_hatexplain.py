import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.metrics import f1_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="models/hatexplain-roberta-mini")
    # Fast debug options to iterate quickly (see comments below)
    ap.add_argument("--fast_debug", action="store_true", help="Use smaller model / fewer epochs for fast iteration")
    ap.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training samples (for quick debug)")
    args = ap.parse_args()

    try:
        ds = load_dataset("Hate-speech-CNERG/hatexplain")
    except NotImplementedError as e:
        # Some environments raise: "Loading a dataset cached in a LocalFileSystem is not supported.".
        # Streaming access can also be blocked. To ensure quick debug runs work in constrained
        # environments, fall back to a small synthetic in-memory dataset that matches the
        # expected fields (post_tokens, annotators.label).
        print("Warning: dataset cache LocalFileSystem not supported; falling back to a small synthetic dataset for debugging.")
        from datasets import Dataset, DatasetDict

        def make_synthetic(n):
            items = []
            for i in range(n):
                # create a tiny token list and cyclic label (0,1,2)
                items.append({
                    "post_tokens": ["this", "is", "sample", f"{i}"],
                    "annotators": {"label": [i % 3]}
                })
            return Dataset.from_list(items)

        ds = DatasetDict({
            "train": make_synthetic(2000),
            "validation": make_synthetic(500),
            "test": make_synthetic(500),
        })

    def to_text(example):
        text = " ".join(example["post_tokens"])
        label = example["annotators"]["label"][0] if isinstance(example["annotators"]["label"], list) else example["annotators"]["label"]
        return {"text": text, "label": int(label)}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    label_names = ["hatespeech","offensive","normal"]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok_fn(batch): return tok(batch["text"], truncation=True)
    ds_tok = ds.map(tok_fn, batched=True)

    # If fast_debug is enabled, override heavy defaults for quick iteration.
    # Changes when --fast_debug is set (documented here):
    # - model: default "roberta-base" (large) -> "distilbert-base-uncased" (smaller/faster)
    # - epochs: default 2 -> 1
    # - batch: default 16 -> 32 (larger batch for throughput if GPU allows)
    # - use fp16 mixed precision if GPU available -> faster training
    if args.fast_debug:
        print("FAST DEBUG: switching to smaller model and fewer epochs for quick iteration")
        # only change model if user kept default
        if args.model == "roberta-base":
            args.model = "distilbert-base-uncased"
        args.epochs = 1
        args.batch = max(1, args.batch * 2)

    # Optionally subset the datasets for very fast debugging/trials
    if args.max_train_samples is not None:
        print(f"Subsetting train set to first {args.max_train_samples} samples for quick debug")
        ds_tok["train"] = ds_tok["train"].select(range(min(args.max_train_samples, len(ds_tok["train"]))))
        # also shrink validation/test to keep predict/debug fast
        ds_tok["validation"] = ds_tok["validation"].select(range(min(200, len(ds_tok["validation"]))))
        ds_tok["test"] = ds_tok["test"].select(range(min(200, len(ds_tok["test"]))))

    metric_acc = evaluate.load("accuracy")
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
        macro_f1 = f1_score(p.label_ids, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)

    # Use fp16 in fast_debug to speed up if CUDA GPU supports it. MPS (Apple) doesn't support fp16 in accelerate.
    fp16_flag = True if (args.fast_debug and torch.cuda.is_available()) else False
    # Build TrainingArguments but filter parameters to those accepted by the installed
    # transformers version to avoid TypeError: unexpected keyword argument.
    import inspect
    desired_args = {
        "output_dir": args.out,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": args.batch,
        "per_device_eval_batch_size": args.batch,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "fp16": fp16_flag,
        "dataloader_num_workers": 4,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    accepted = set(sig.parameters.keys())
    # remove 'self' if present
    accepted.discard("self")
    filtered = {k: v for k, v in desired_args.items() if k in accepted}
    dropped = [k for k in desired_args.keys() if k not in filtered]
    if dropped:
        print(f"TrainingArguments: dropped unsupported args for this transformers version: {dropped}")
    # If the evaluation strategy was dropped, loading the best model at end will fail
    if "evaluation_strategy" not in filtered and filtered.get("load_best_model_at_end", False):
        print("Evaluation strategy unsupported; disabling load_best_model_at_end to avoid TrainingArguments error.")
        filtered["load_best_model_at_end"] = False
    args_tr = TrainingArguments(**filtered)


    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    preds = trainer.predict(ds_tok["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(-1)
    print(classification_report(y_true, y_pred, target_names=label_names))

if __name__ == "__main__":
    main()
