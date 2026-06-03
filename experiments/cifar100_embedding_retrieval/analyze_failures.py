from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifar100_meta import COARSE_CLASS_NAMES
from data import make_test_dataset, make_test_image_dataset, make_train_image_dataset
from model import build_model
from train_classifier import collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--run-dir", default="runs/cifar100_embedding_retrieval")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--max-failures", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--report-name", default="failure_reports")
    parser.add_argument("--no-grids", action="store_true")
    parser.add_argument("--support-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def nearest_from_mask(similarities, mask, count):
    masked = similarities.clone()
    masked[~mask] = -float("inf")
    values, positions = torch.topk(masked, k=min(count, int(mask.sum().item())))
    return positions.cpu(), values.cpu()


def tensor_to_image(image_tensor):
    return image_tensor.permute(1, 2, 0).numpy()


def draw_failure_grid(record, test_image_dataset, train_image_dataset, output_path):
    query = test_image_dataset[record["test_index"]]
    groups = [
        (
            "Query image",
            f"true: {record['true_class']}\npred: {record['predicted_class']}",
            [{"dataset_index": record["test_index"], "distance": None, "class_name": record["true_class"]}],
        ),
        ("Global nearest", "all support classes", record["global_neighbors"]),
        (
            "Predicted-class nearest",
            f"only {record['predicted_class']}",
            record["predicted_class_neighbors"],
        ),
        ("True-class nearest", f"only {record['true_class']}", record["true_class_neighbors"]),
    ]
    rows = len(groups)
    image_cols = max(len(items) for _, _, items in groups)
    cols = image_cols + 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.2, rows * 2.4),
        gridspec_kw={"width_ratios": [0.95] + [1] * image_cols},
    )
    if rows == 1:
        axes = [axes]

    for row, (title, description, items) in enumerate(groups):
        label_ax = axes[row][0] if rows > 1 else axes[0]
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            f"{title}\n{description}",
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            wrap=True,
        )
        for col in range(cols):
            if col == 0:
                continue
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis("off")
            item_index = col - 1
            if item_index >= len(items):
                continue
            item = items[item_index]
            if title == "Query image":
                sample = query
                subtitle = ""
            else:
                sample = train_image_dataset[item["dataset_index"]]
                subtitle = f"{item['class_name']}\nd={item['distance']:.3f}"
            ax.imshow(tensor_to_image(sample["image"]))
            if subtitle:
                ax.set_title(subtitle, fontsize=8)

    fig.suptitle(
        f"confidence={record['predicted_confidence']:.3f}, margin={record['similarity_margin']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    report_dir = run_dir / args.report_name
    image_dir = report_dir / "grids"
    image_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(run_dir / "best_classifier.pt", map_location="cpu")
    index = torch.load(run_dir / "support_index.pt", map_location="cpu")
    classes = index["classes"]

    test_dataset, _ = make_test_dataset(args.data_root)
    test_image_dataset, _ = make_test_image_dataset(args.data_root)
    train_image_dataset, _ = make_train_image_dataset(args.data_root)

    loader = DataLoader(
        test_dataset,
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

    support_embeddings = index["embeddings"].to(device)
    support_labels = index["fine_labels"].to(device)
    support_indices = index["indices"]

    output_jsonl = report_dir / "failures.jsonl"
    failure_count = 0
    margins = []
    with output_jsonl.open("w") as f:
        for batch in tqdm(loader, desc="analyzing failures"):
            images = batch["image"].to(device)
            labels = batch["fine"].to(device)
            coarse_labels = batch["coarse"].to(device)
            logits, embeddings = model(images, return_embedding=True)
            probabilities = logits.softmax(dim=1)
            top_values, top_indices = torch.topk(probabilities, k=args.top_k, dim=1)
            predictions = top_indices[:, 0]
            normalized = F.normalize(embeddings, dim=1)
            similarities = normalized @ support_embeddings.T

            for row in range(images.size(0)):
                true_label = int(labels[row].item())
                predicted_label = int(predictions[row].item())
                if predicted_label == true_label:
                    continue
                predicted_confidence = float(top_values[row, 0].item())
                if predicted_confidence < args.min_confidence:
                    continue

                sims = similarities[row]
                pred_positions, pred_sims = nearest_from_mask(
                    sims,
                    support_labels == predicted_label,
                    args.neighbors,
                )
                true_positions, true_sims = nearest_from_mask(
                    sims,
                    support_labels == true_label,
                    args.neighbors,
                )
                global_sims, global_positions = torch.topk(sims.cpu(), k=args.neighbors)

                pred_distance = float(1.0 - pred_sims[0].item())
                true_distance = float(1.0 - true_sims[0].item())
                margin = true_distance - pred_distance
                margins.append(margin)

                def make_neighbors(positions, sims_for_positions):
                    rows = []
                    for pos, sim in zip(positions.tolist(), sims_for_positions.tolist()):
                        fine = int(index["fine_labels"][pos].item())
                        coarse = int(index["coarse_labels"][pos].item())
                        rows.append(
                            {
                                "support_position": pos,
                                "dataset_index": int(support_indices[pos].item()),
                                "class_id": fine,
                                "class_name": classes[fine],
                                "coarse_id": coarse,
                                "coarse_name": COARSE_CLASS_NAMES[coarse],
                                "similarity": float(sim),
                                "distance": float(1.0 - sim),
                            }
                        )
                    return rows

                record = {
                    "test_index": int(batch["index"][row].item()),
                    "true_class_id": true_label,
                    "true_class": classes[true_label],
                    "true_coarse_id": int(coarse_labels[row].item()),
                    "true_coarse": COARSE_CLASS_NAMES[int(coarse_labels[row].item())],
                    "predicted_class_id": predicted_label,
                    "predicted_class": classes[predicted_label],
                    "predicted_coarse_id": int(index["coarse_labels"][int(pred_positions[0].item())].item()),
                    "predicted_confidence": predicted_confidence,
                    "top_k": [
                        {
                            "class_id": int(class_id.item()),
                            "class_name": classes[int(class_id.item())],
                            "confidence": float(conf.item()),
                        }
                        for conf, class_id in zip(top_values[row], top_indices[row])
                    ],
                    "predicted_class_nearest_distance": pred_distance,
                    "true_class_nearest_distance": true_distance,
                    "similarity_margin": margin,
                    "global_neighbors": make_neighbors(global_positions, global_sims),
                    "predicted_class_neighbors": make_neighbors(pred_positions, pred_sims),
                    "true_class_neighbors": make_neighbors(true_positions, true_sims),
                }
                grid_path = image_dir / f"failure_{failure_count:04d}_{record['true_class']}_as_{record['predicted_class']}.png"
                record["grid_path"] = str(grid_path)
                if not args.no_grids:
                    draw_failure_grid(record, test_image_dataset, train_image_dataset, grid_path)
                f.write(json.dumps(record) + "\n")
                failure_count += 1
                if failure_count >= args.max_failures:
                    break
            if failure_count >= args.max_failures:
                break

    positive_margins = sum(1 for margin in margins if margin > 0)
    summary = {
        "failures_analyzed": failure_count,
        "min_confidence": args.min_confidence,
        "positive_similarity_margin_count": positive_margins,
        "positive_similarity_margin_fraction": positive_margins / max(1, len(margins)),
        "mean_similarity_margin": sum(margins) / max(1, len(margins)),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
