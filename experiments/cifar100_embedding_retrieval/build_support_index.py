from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import make_train_support_datasets
from model import build_model
from train_classifier import collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-dir", default="runs/cifar100_embedding_retrieval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--support-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = torch.load(run_dir / "best_classifier.pt", map_location="cpu")

    _, support_dataset, classes = make_train_support_datasets(
        args.data_root,
        split_path=run_dir / "split.json",
        support_fraction=args.support_fraction,
        seed=args.seed,
    )
    loader = DataLoader(
        support_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    device = torch.device(args.device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    embeddings = []
    fine_labels = []
    coarse_labels = []
    indices = []
    for batch in tqdm(loader, desc="embedding support"):
        images = batch["image"].to(device)
        _, embedding = model(images, return_embedding=True)
        embeddings.append(F.normalize(embedding, dim=1).cpu())
        fine_labels.append(batch["fine"])
        coarse_labels.append(batch["coarse"])
        indices.append(batch["index"])

    torch.save(
        {
            "embeddings": torch.cat(embeddings, dim=0),
            "fine_labels": torch.cat(fine_labels, dim=0),
            "coarse_labels": torch.cat(coarse_labels, dim=0),
            "indices": torch.cat(indices, dim=0),
            "classes": classes,
            "metric": "cosine_distance",
        },
        run_dir / "support_index.pt",
    )
    print(f"saved {run_dir / 'support_index.pt'}")


if __name__ == "__main__":
    main()
