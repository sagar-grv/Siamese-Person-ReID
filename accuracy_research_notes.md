# Accuracy improvement research notes

## Current baseline

The current checkpoint reports 200 queries, 854 gallery images, 58.5% top-1 accuracy, 82.5% top-5 accuracy, and 0.3296 mAP on the Market-1501 subset. The current encoder is a compact CNN with a 128-dimensional normalized embedding, trained for 12 epochs with 1,200 deterministic pairs per epoch and a basic contrastive loss.

## Paper findings

- Luo et al., *A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification* (arXiv:1906.08332), describes a strong ResNet-50 baseline with BNNeck separating metric and classification feature spaces. The paper reports 94.5% rank-1 and 85.9% mAP on Market-1501 for its full baseline, which is not directly comparable to the current small subset result but motivates BNNeck, identity classification, label smoothing, warmup, random erasing, and stronger backbones.
- He et al., *TransReID: Transformer-based Object Re-Identification* (arXiv:2102.04378), motivates patch-based transformer features and camera/viewpoint side-information embeddings. This is a later, heavier upgrade path rather than the first change for this compact demo.

## Recommendation direction

The first improvements should be: use the full Market-1501 train split and official query/gallery protocol; replace random pair sampling with identity-balanced batches and hard/semi-hard negative mining; initialize a stronger pretrained backbone; combine identity classification loss with batch-hard triplet loss or contrastive loss; add realistic person-ReID augmentations; use camera-aware sampling and camera-balanced evaluation; train longer with warmup and a tuned learning-rate schedule; and tune retrieval post-processing such as query expansion only after the baseline is stable. Evaluation should report CMC rank-1/rank-5/rank-10, mAP, per-camera performance, and confidence/calibration rather than relying on a single top-1 score.

## Sources

- https://arxiv.org/abs/1906.08332
- https://arxiv.org/abs/2102.04378
- https://arxiv.org/abs/1502.02171
