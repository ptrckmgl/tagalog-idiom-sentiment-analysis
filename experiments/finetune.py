"""
Fine-tune one encoder model on one task (figurative/literal or sentiment).

How it works:
1. Reads a labeled CSV with columns `idioms`, `text` and the label column of the task.
2. Uses the `split` column (train / val / test_seen / test_unseen) if there is one.
   Without it, a random stratified 80/10/10 split by sentence is made. That is fine for
   a prototype but NOT the thesis design (split by idiom, grouped by document).
3. Feeds (idiom, sentence) pairs to the model, trains with the Hugging Face Trainer and
   keeps the epoch with the best validation macro-F1.
4. Scores every test set, saves the predictions and appends one row to results.csv.

Untested on real data. Needs a GPU for any real run.

Usage:
    python experiments/finetune.py --data labeled.csv --task figurative
    python experiments/finetune.py --data labeled.csv --task sentiment --model xlm-roberta-base --lr 3e-5 --seed 43
    python experiments/finetune.py --data labeled.csv --task figurative --batch-size 8 --grad-accum 2

Output:
    runs/<model>_<task>_lr<lr>_s<seed>/predictions_<test set>.csv
    runs/results.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# ==============================================================================
# SETTINGS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

DEFAULT_MODEL = "jcblaise/roberta-tagalog-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42
SPLIT_SEED = 67

# label column and class order for each task
TASKS = {
    "figurative": ("figurative", ["literal", "figurative"]),
    "sentiment": ("sentiment", ["negative", "neutral", "positive"]),
}
TEST_SETS = ["test_seen", "test_unseen"]


# ==============================================================================
# 1. Data
# ==============================================================================

def load_splits(path, label_col, classes):
    """Return {split name: DataFrame} with an integer `labels` column."""

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df[label_col] = df[label_col].str.strip().str.lower()
    df = df[df[label_col] != ""]

    unknown = sorted(set(df[label_col]) - set(classes))
    if unknown:
        raise SystemExit(f"Unknown {label_col} labels {unknown}; expected {classes}")
    df["labels"] = df[label_col].map({c: i for i, c in enumerate(classes)})

    if "split" in df.columns:
        return {name: part for name, part in df.groupby("split")}

    print("No `split` column: making a random 80/10/10 split by sentence (prototype only).")
    train, rest = train_test_split(df, test_size=0.2, stratify=df["labels"], random_state=SPLIT_SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["labels"], random_state=SPLIT_SEED)
    return {"train": train, "val": val, "test_seen": test}


# ==============================================================================
# 2. Training
# ==============================================================================

def compute_metrics(pred):
    """Macro-F1 is the primary metric."""

    preds = np.argmax(pred.predictions, axis=-1)
    return {"macro_f1": f1_score(pred.label_ids, preds, average="macro")}


def train_and_score(args, splits, classes):
    """Fine-tune, then score each test set. Returns a dict for results.csv."""

    # imported here so --help works without torch installed
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(df):
        data = Dataset.from_pandas(df[["idioms", "text", "labels"]].reset_index(drop=True))
        return data.map(
            lambda b: tokenizer(b["idioms"], b["text"], truncation=True, max_length=args.max_length),
            batched=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(classes))

    run_name = f"{args.model.split('/')[-1]}_{args.task}_lr{args.lr}_s{args.seed}"
    run_dir = os.path.join(args.output, run_name)

    training_args = TrainingArguments(
        output_dir=run_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=not args.no_fp16,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encode(splits["train"]),
        eval_dataset=encode(splits["val"]),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    row = {
        "model": args.model,
        "task": args.task,
        "lr": args.lr,
        "seed": args.seed,
        "val_macro_f1": trainer.evaluate()["eval_macro_f1"],
    }

    for name in TEST_SETS:
        if name not in splits:
            continue
        test = splits[name]
        logits = trainer.predict(encode(test)).predictions
        preds = np.argmax(logits, axis=-1)
        row[f"{name}_macro_f1"] = f1_score(test["labels"], preds, average="macro")
        print(f"\n{name}\n" + classification_report(
            test["labels"], preds, labels=range(len(classes)), target_names=classes, zero_division=0
        ))
        out = test[["idioms", "text", "labels"]].copy()
        out["predicted"] = preds
        out.to_csv(os.path.join(run_dir, f"predictions_{name}.csv"), index=False, encoding="utf-8-sig")

    return row


# ==============================================================================
# 3. Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="labeled CSV (idioms, text, label column)")
    parser.add_argument("--task", required=True, choices=TASKS, help="which label to learn")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="learning rate")
    parser.add_argument("--seed", type=int, default=SEED, help="training seed")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="maximum epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="batch size per step")
    parser.add_argument("--grad-accum", type=int, default=1, help="steps to add up before updating")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="max tokens per pair")
    parser.add_argument("--no-fp16", action="store_true", help="turn off half precision (needed on CPU)")
    parser.add_argument("--output", default=RUNS_DIR, help="folder for runs and results.csv")
    args = parser.parse_args()

    label_col, classes = TASKS[args.task]
    splits = load_splits(args.data, label_col, classes)
    for need in ("train", "val"):
        if need not in splits:
            parser.error(f"the data has no `{need}` split")
    print({name: len(part) for name, part in splits.items()})

    row = train_and_score(args, splits, classes)

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "results.csv")
    pd.DataFrame([row]).to_csv(
        results_path, mode="a", header=not os.path.exists(results_path), index=False, encoding="utf-8-sig"
    )
    print(f"\nAppended to {results_path}: {row}")


if __name__ == "__main__":
    main()
