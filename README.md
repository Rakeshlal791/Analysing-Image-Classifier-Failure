# Understanding High-Confidence Classifier Failures

This project studies why image classifiers make **confidently wrong predictions**. Instead of only reporting that a classifier failed, we ask whether the failed image is visually close to examples from the class the model predicted.

The core idea:

> A high-confidence mistake may happen because the input is closer to the predicted class than the true class in the classifier's own embedding space.

## Hero Examples

The grids below show two high-confidence CIFAR-100 failures. Each row retrieves nearest examples from a 5% support set using the classifier's penultimate-layer embedding.

| Clock predicted as lamp | Dinosaur predicted as aquarium fish |
| --- | --- |
| ![Clock predicted as lamp](runs/cifar100_embedding_retrieval/high_confidence_failure_reports/grids/failure_0012_clock_as_lamp.png) | ![Dinosaur predicted as aquarium fish](runs/cifar100_embedding_retrieval/high_confidence_failure_reports/grids/failure_0013_dinosaur_as_aquarium_fish.png) |

In the clock example, the classifier predicts `lamp` with `0.980` confidence. The nearest lamp support image is much closer than the nearest clock support image:

```txt
nearest predicted-class distance: 0.124
nearest true-class distance:      0.239
similarity margin:                0.115
```

In the dinosaur example, the classifier predicts `aquarium_fish` with `0.996` confidence:

```txt
nearest predicted-class distance: 0.094
nearest true-class distance:      0.184
similarity margin:                0.090
```

These are not just wrong predictions. They are wrong predictions where the classifier's own representation places the image closer to the predicted class than the true class.

## Research Question

When a classifier is very confidently wrong, is the mistake related to visual similarity in the model's learned representation?

More concretely:

```txt
true class:      clock
predicted class: lamp
confidence:      0.980
```

Do the nearest support examples in the classifier embedding look like lamps or clocks?

## Dataset

The experiment uses **CIFAR-100**, a 60,000-image benchmark with 100 fine-grained classes and 20 superclasses. Each class has 500 training images and 100 test images. Images are small, `32 x 32` RGB examples, which makes this a difficult setting for fine-grained visual explanations.

Split used here:

```txt
training split: 47,500 images, 95% of CIFAR-100 train
support split:   2,500 images, 5% of CIFAR-100 train
test split:     10,000 images
```

The support split is stratified, so each class contributes about 25 examples to the retrieval database.

References: [TensorFlow Datasets CIFAR-100](https://tensorflow.google.cn/datasets/catalog/cifar100), [CIFAR-100 dataset card](https://huggingface.co/datasets/cifar100/blob/main/README.md?code=true).

## Model

The classifier is a pretrained **CIFAR-100 ResNet-56**.

For prediction, the model does the normal classification path:

```txt
image -> ResNet-56 backbone -> penultimate embedding -> linear classifier -> logits
```

For explanation, we reuse the same penultimate embedding:

```txt
image -> ResNet-56 backbone -> penultimate embedding
```

This is important: retrieval does not use a separate CLIP model, contrastive model, or hand-engineered similarity function. The retrieved examples are neighbors according to the classifier's own representation.

This follows the spirit of Deep k-Nearest Neighbors, where examples are compared in learned neural-network representations to make model behavior more interpretable: [Deep k-NN](https://arxiv.org/abs/1803.04765).

## Method

1. Fine-tune the pretrained CIFAR-100 classifier on the 95% train split.
2. Hold out a 5% support set from the training data.
3. Embed every support image using the classifier's penultimate layer.
4. During test evaluation, find high-confidence wrong predictions.
5. For each failure, retrieve nearest support images from:
   - all support classes,
   - the predicted class only,
   - the true class only.
6. Save a JSONL record and a visual grid for each failure.

Distances are computed using cosine distance over normalized embeddings:

```txt
cosine_distance = 1 - cosine_similarity(test_embedding, support_embedding)
```

The key metric is:

```txt
similarity_margin = true_class_nearest_distance - predicted_class_nearest_distance
```

Interpretation:

```txt
similarity_margin > 0
```

means the failed image is closer to the predicted class than the true class in the classifier's embedding space.

## Findings

The most interesting slice is **high-confidence failure**, where the classifier is wrong with predicted probability at least `0.9`.

Current high-confidence result:

```txt
confidence threshold:                 >= 0.9
high-confidence failures analyzed:       957
positive similarity-margin failures:     850
positive similarity-margin fraction:   88.8%
mean similarity margin:                0.0595
```

Main finding:

> In 88.8% of high-confidence mistakes, the image is closer to examples from the predicted class than examples from the true class in the classifier's own embedding space.

This suggests that many confident errors are not arbitrary. They often reflect the geometry of the learned representation: the model is confidently wrong because the image is internally represented as more similar to the wrong class.

The remaining 11.2% are equally important. These are high-confidence failures where the nearest true-class support example is closer than the nearest predicted-class support example:

```txt
non-positive-margin failures: 107 / 957
same-superclass cases:        62 / 107 = 57.9%
mean confidence:              0.954
mean margin:                 -0.021
```

These cases show that nearest-neighbor similarity is not a complete explanation of classifier failure. Many are still semantically plausible confusions, such as `plate -> bowl`, `crocodile -> lizard`, `tiger -> leopard`, and `crab -> lobster`, but the final classifier head disagrees with the nearest-neighbor evidence in embedding space.

This gives two useful failure categories:

```txt
Positive-margin failures:
the wrong prediction is supported by local embedding similarity.

Non-positive-margin failures:
the wrong prediction is not supported by nearest-neighbor similarity;
these may reflect decision-boundary, classifier-head, calibration, or support-set effects.
```

## How To Read A Failure Grid

Each grid has four rows:

```txt
Query image
Global nearest support examples
Predicted-class nearest examples
True-class nearest examples
```

Items within a row are sorted left to right by increasing distance. Smaller `d=` means closer in embedding space.

For a strong visual-similarity failure, we expect:

```txt
nearest predicted-class distance < nearest true-class distance
```

That is exactly what happens in both hero examples.

## Interpretation

This experiment explains failures at the **representation level**.

It does not prove a causal chain like:

```txt
the model saw this exact support image, therefore it predicted this class
```

Instead, it supports a more careful claim:

```txt
the failed image lies closer to predicted-class examples than true-class examples
in the same embedding space the classifier uses for prediction
```

Some failures are semantically intuitive, such as `clock -> lamp`: the clock has a bright circular face that resembles glowing lamp examples. Others reveal spurious similarity, such as texture, color, background, or thin edge patterns.

## Outputs

General failure reports:

```txt
runs/cifar100_embedding_retrieval/failure_reports/
```

High-confidence failure reports:

```txt
runs/cifar100_embedding_retrieval/high_confidence_failure_reports/
```

Important files:

```txt
summary.json      aggregate metrics
failures.jsonl    per-failure records
grids/            visual retrieval grids
```

## Run

Use the `prg` conda environment:

```bash
/home/rakesh/miniconda3/envs/prg/bin/python experiments/cifar100_embedding_retrieval/train_classifier.py
/home/rakesh/miniconda3/envs/prg/bin/python experiments/cifar100_embedding_retrieval/build_support_index.py
```

Analyze general failures:

```bash
/home/rakesh/miniconda3/envs/prg/bin/python experiments/cifar100_embedding_retrieval/analyze_failures.py
```

Analyze high-confidence failures:

```bash
/home/rakesh/miniconda3/envs/prg/bin/python experiments/cifar100_embedding_retrieval/analyze_failures.py \
  --min-confidence 0.9 \
  --report-name high_confidence_failure_reports \
  --max-failures 1000
```

For faster threshold sweeps without rendering PNG grids:

```bash
/home/rakesh/miniconda3/envs/prg/bin/python experiments/cifar100_embedding_retrieval/analyze_failures.py \
  --min-confidence 0.9 \
  --report-name high_confidence_failure_reports_fast \
  --max-failures 1000 \
  --no-grids
```

## Resume Summary

Built an embedding-space failure analysis pipeline for CIFAR-100 classifiers using the classifier's own penultimate representation. The system retrieves nearest support-set examples for high-confidence misclassifications and found that **88.8% of wrong predictions with confidence >= 0.9 were closer to predicted-class examples than true-class examples**, suggesting representation-level visual similarity as a major driver of confident classifier errors.

## Next Steps

- Add baselines using pixel similarity and external embeddings.
- Compare high-confidence failures across multiple random support splits.
- Annotate failure types: semantic similarity, texture bias, background bias, shape bias, and superclass confusion.
- Add per-class and per-superclass failure tables.
- Compare retrieval explanations with saliency methods such as Grad-CAM.
