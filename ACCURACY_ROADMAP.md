# Accuracy Roadmap for Siamese Person Re-Identification

## Current baseline

The current model uses a compact CNN encoder with a 128-dimensional L2-normalized embedding and a basic contrastive loss. It trains on 1,200 deterministic pairs per epoch for 12 epochs and evaluates on a small identity-disjoint Market-1501 subset.

The recorded baseline is:

| Metric | Baseline |
|---|---:|
| Query images | 200 |
| Gallery images | 854 |
| Rank-1 / top-1 | 58.5% |
| Rank-5 | 82.5% |
| mAP | 0.3296 |

These results are useful for validating the pipeline, but they should not be compared directly with full-dataset research numbers because the current model, training duration, and subset protocol are much smaller.

## Highest-impact improvements

### 1. Train on the full official protocol

The current subset limits both identity diversity and camera variation. The first experiment should use the complete Market-1501 training split and the official query/gallery evaluation protocol. Keep identities disjoint between training and evaluation, and report rank-1, rank-5, rank-10, mAP, and per-camera results.

### 2. Replace random pairs with identity-balanced batches

`PairDataset` currently samples one positive or negative pair per index. This wastes the structure of each minibatch and produces weak negatives. Use a person-balanced sampler, for example 8 identities × 4 images per identity per batch. Within each batch, compute all pairwise distances and use batch-hard or soft-margin triplet mining. This gives the model difficult same-identity and different-identity examples instead of mostly random negatives.

### 3. Add identity classification to the metric objective

The current network learns only pair distances. Add a classification head over training identities and optimize a joint objective:

```text
L = L_cross_entropy + λ · L_batch_hard_triplet
```

Use the normalized embedding for retrieval and a separate BatchNorm neck or classification feature for the identity classifier. This follows the strong-baseline observation that classification and metric objectives benefit from separate feature spaces [1].

### 4. Start from a pretrained backbone

The custom four-layer CNN is compact, but it has limited capacity and is trained from scratch. Replace it with a pretrained ResNet-50 or an efficient OSNet-style backbone, remove its ImageNet classifier, and project its pooled feature to 256 or 512 dimensions. Freeze the backbone for the first few epochs, train the projection and classifier, then unfreeze with a smaller learning rate.

### 5. Use person-ReID augmentations

Add random resized crop with conservative scale limits, horizontal flip, color jitter, random erasing, and mild blur. Avoid aggressive crops that remove the identity-bearing body region. Random erasing is particularly useful for simulating occlusion, while color changes improve robustness across cameras.

### 6. Train longer with a staged schedule

Twelve epochs with 1,200 pairs per epoch is appropriate for a smoke test, not for a strong benchmark. A practical next run is 60–120 epochs with a short warmup, AdamW or SGD with momentum, cosine decay or a step schedule, and checkpoint selection based on validation mAP rather than training loss alone. Save the best validation checkpoint and an early-stopping patience window.

### 7. Make camera variation explicit

Market-1501 contains multiple camera views. During training, prefer positive pairs from different cameras when available and monitor same-camera versus cross-camera retrieval separately. A later upgrade can add camera/viewpoint side information or camera-adversarial regularization. TransReID demonstrates the value of modelling camera and viewpoint variation as side information [2].

### 8. Improve the retrieval decision, not only the embedding

After the embedding is strong, evaluate cosine similarity thresholds on a validation split. Calibrate a `same identity` confidence score instead of presenting every nearest neighbor as a certain match. You can also evaluate query expansion or k-reciprocal re-ranking, but these should be added only after the raw embedding baseline is stable and always reported separately.

## Recommended experiment order

| Stage | Change | Why first |
|---|---|---|
| A | Full Market-1501 protocol, 60+ epochs, stronger augmentations | Removes the largest data and training limitations. |
| B | Identity-balanced batches with batch-hard triplet loss | Improves negative quality and uses minibatches efficiently. |
| C | Pretrained ResNet-50 or OSNet-style encoder | Increases representation capacity and transfer quality. |
| D | Classification head + BNNeck + label smoothing | Adds identity supervision while separating classifier and metric features. |
| E | Camera-aware positive sampling and per-camera analysis | Targets the core cross-camera failure mode. |
| F | Transformer backbone or TransReID-style camera/view modules | Advanced upgrade after the CNN baseline is reliable. |

## Evaluation safeguards

Do not select a checkpoint solely by pairwise training loss. Track validation mAP and CMC rank metrics after every epoch. Include a no-match or low-confidence outcome for queries whose best score falls below the validation threshold. Report mean and standard deviation across at least three random seeds for meaningful comparisons, and keep the test identities untouched until the final evaluation.

For this project, the most valuable first implementation is **full-data training plus identity-balanced batch-hard triplet learning with a pretrained backbone**. That combination should be attempted before moving to transformers or complex architectural modules.

## References

[1]: https://arxiv.org/abs/1906.08332 "A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification"
[2]: https://arxiv.org/abs/2102.04378 "TransReID: Transformer-based Object Re-Identification"
[3]: https://arxiv.org/abs/1502.02171 "Person Re-identification Meets Image Search"
