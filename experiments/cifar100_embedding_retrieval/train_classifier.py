from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_test_dataset, make_train_support_datasets
from model import build_model


def collate(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "fine": torch.tensor([item["fine"] for item in batch], dtype=torch.long),
        "coarse": torch.tensor([item["coarse"] for item in batch], dtype=torch.long),
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["fine"].to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return {"loss": total_loss / total, "accuracy": correct / total}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-dir", default="runs/cifar100_embedding_retrieval")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--support-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    split_path = run_dir / "split.json"

    train_dataset, support_dataset, classes = make_train_support_datasets(
        args.data_root,
        split_path=split_path,
        support_fraction=args.support_fraction,
        seed=args.seed,
    )
    test_dataset, _ = make_test_dataset(args.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    support_loader = DataLoader(
        support_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    device = torch.device(args.device)
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["fine"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)
            pbar.set_postfix(
                loss=train_loss / train_total,
                acc=train_correct / train_total,
            )

        scheduler.step()
        support_metrics = evaluate(model, support_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "support_loss": support_metrics["loss"],
            "support_accuracy": support_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record, indent=2))

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "args": vars(args),
                    "epoch": epoch,
                    "test_accuracy": test_metrics["accuracy"],
                },
                run_dir / "best_classifier.pt",
            )


if __name__ == "__main__":
    main()
