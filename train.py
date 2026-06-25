import argparse, os, json
import torch
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.classifier import TextClassifier
from data.dataset import TextDataset

def train(args):
    df = pd.read_csv(args.data)
    labels = sorted(df["label"].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for l, i in label2id.items()}

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    clf = TextClassifier(model_name=args.model, num_labels=len(labels))
    clf.label2id = label2id; clf.id2label = id2label

    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].map(label2id).tolist(), clf.tokenizer, args.max_length)
    val_ds   = TextDataset(val_df["text"].tolist(),   val_df["label"].map(label2id).tolist(),   clf.tokenizer, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    optimizer = AdamW(clf.model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dl)*args.epochs//10,
                                                num_training_steps=len(train_dl)*args.epochs)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        clf.model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}"):
            out = clf.model(**{k: v.to(clf.device) for k, v in batch.items()})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        clf.model.eval(); correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                out = clf.model(**{k: v.to(clf.device) for k, v in batch.items()})
                preds = torch.argmax(out.logits, dim=-1)
                correct += (preds == batch["labels"].to(clf.device)).sum().item()
                total   += batch["labels"].size(0)
        acc = correct / total
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best:
            best = acc; clf.save(args.output_dir)
            with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f)
    print(f"\n✅ Best val_acc: {best:.4f}. Saved to {args.output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample.csv")
    p.add_argument("--model", default="bert-base-multilingual-cased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--output_dir", default="./saved_model")
    args = p.parse_args(); os.makedirs(args.output_dir, exist_ok=True); train(args)
