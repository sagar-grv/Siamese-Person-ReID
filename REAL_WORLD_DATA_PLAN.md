# Real-World Data and Deployment Plan

## Product direction

The safest and most useful first product is an authorized visual-retrieval tool for retrospective review inside one controlled environment. An operator submits a person crop from an approved incident, retrieves visually similar gallery views, and reviews the evidence with camera and time metadata. The model should be treated as a search assistant, not as proof of identity.

## 1. Build a site-specific dataset

Market-1501 is useful for bootstrapping, but it does not represent the target cameras, lighting, compression, clothing distribution, or operating procedures. Collect a small site-specific dataset under written authorization and with the appropriate consent, notice, retention, and access controls for the target jurisdiction. Do not mix data from different purposes without documenting the change in purpose.

Use a manifest rather than relying only on filenames:

| Field | Purpose |
|---|---|
| `image_path` | Stable location of the person crop. |
| `person_id` | Pseudonymous identity label used only for training/evaluation. |
| `camera_id` | Camera-domain identifier. |
| `session_id` | Visit or capture-session identifier. |
| `timestamp` | Time bucket for temporal split and drift analysis. |
| `bbox_quality` | Quality flag for blur, truncation, occlusion, or detector failure. |
| `consent_status` | Authorization state for the sample. |

Prioritize variation that causes failures in the current benchmark: different cameras, indoor/outdoor scenes, day/night lighting, occlusions, crowd density, body pose, image compression, and clothing changes. Include hard negative pairs where people have similar clothing or body shape.

## 2. Split the data without leakage

Do not randomly split adjacent frames. A random frame split can place near-duplicate images from the same camera sequence in both training and test sets and produce an inflated score. Split by identity for closed-set generalization tests, and also create a deployment-style split by session or day. Keep a final test period untouched until the end.

| Split | Recommended role |
|---|---|
| Development train | Fine-tune the encoder and classifier. |
| Validation | Tune thresholds, mining, augmentation, and checkpoint selection. |
| Final test period | One-time estimate of expected deployment performance. |
| Stress set | Occlusion, low light, blur, crowded scenes, and camera-specific failures. |

For each query, evaluate cross-camera retrieval separately from same-camera retrieval. Report results per camera and per condition, not only one global score.

## 3. Train in stages

Start from the current pretrained ResNet-18 pipeline. First train the projection and classification layers with the backbone frozen for a few epochs. Then unfreeze the last ResNet blocks with a smaller learning rate. Keep identity-balanced `P × K` batches, batch-hard triplet mining, label smoothing, random erasing, and camera-aware positive sampling.

A practical objective is:

```text
L = L_cross_entropy + λ_triplet L_batch_hard_triplet
```

Do not add many losses at once. Establish a clean baseline, then run one controlled change at a time. If target labels are limited, use Market-1501 or other licensed ReID data for pretraining, then fine-tune on the authorized site data. Unsupervised pseudo-labeling can help later, but pseudo-label errors must be measured and manually audited before being used for training.

The strongest next code changes are:

| Repository change | Purpose |
|---|---|
| `data/site/manifest.csv` | Store site-specific paths, pseudonymous IDs, cameras, sessions, quality, and consent state. |
| `train_upgraded.py --manifest ...` | Train from a manifest instead of hard-coded Market folders. |
| `IdentityCameraSampler` | Prefer positive pairs from different cameras and keep identities balanced. |
| `evaluate_real_world.py` | Produce per-camera, per-session, stress-set, CMC, mAP, and threshold metrics. |
| `calibrate_threshold.py` | Select match/review/no-match thresholds on validation data only. |
| `gallery_version.json` | Tie every deployed embedding index to a model, dataset, and evaluation report. |

## 4. Evaluate the operating point

Top-1 accuracy alone is not enough for an operational system. Measure rank-1, rank-5, rank-10, mAP, false-match rate, miss rate, and latency. Choose a threshold based on the cost of false matches versus missed matches in the intended workflow. Always include an explicit `no reliable match` result.

Create an error-review table with the query, top candidates, camera pair, score, true relationship, failure category, and reviewer decision. Use that table to drive the next training set rather than adding data randomly.

## 5. Reduce the domain gap

A model trained on one benchmark can fail when camera optics, weather, compression, or scene layout change. Domain-generalizable ReID research specifically targets this deployment problem [1]. Practical steps are camera-balanced sampling, style and illumination augmentation, camera-specific validation, and fine-tuning on a small labeled target set. A later advanced path can use domain-invariant feature learning or unsupervised adaptation, but it should not replace a clean labeled validation protocol.

For larger systems, store gallery embeddings in a private vector index and keep metadata in a separate access-controlled database. FAISS is suitable for an on-premise prototype; a managed vector database may be appropriate only after access, retention, and encryption requirements are defined. Use the browser-only demo for local experimentation, not as the default architecture for sensitive production data.

## 6. Add responsible product controls

Apply risk management, transparency, evaluation, and human oversight throughout the lifecycle, consistent with the NIST AI Risk Management Framework [2]. The interface should show the model version, gallery version, score, camera, timestamp, and an uncertainty label. It should require a human review action before an incident is marked as resolved.

The production service should include authentication, role-based access, encrypted transport and storage, short retention, deletion workflows, audit logs, and rate limits. Prohibit continuous public monitoring, automatic access denial, employment decisions, law-enforcement identification, and other high-impact decisions. Obtain current jurisdiction-specific privacy and legal advice before collecting or deploying real-world person data.

## 7. A practical implementation sequence

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 1. Data foundation | Site manifest, quality labels, consent/authorization record, and session-aware splits | No split leakage; data provenance is auditable. |
| 2. Target fine-tuning | ResNet-18 fine-tuning with camera-aware `P × K` batches | Validation mAP improves without stress-set collapse. |
| 3. Thresholding | Match/review/no-match calibration | Thresholds are selected only on validation data. |
| 4. Evaluation dashboard | Per-camera and per-condition metrics, error review, latency | Operators can see failure modes instead of one opaque score. |
| 5. Private pilot | Authenticated review UI, versioned gallery, audit log | Human reviewers can explain and override every result. |
| 6. Controlled rollout | Monitoring, drift checks, retraining cadence, incident process | Deployment has rollback and data-deletion procedures. |

## References

[1]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Song_Generalizable_Person_Re-Identification_by_Domain-Invariant_Mapping_Network_CVPR_2019_paper.pdf "Generalizable Person Re-identification by Domain-Invariant Mapping Network"
[2]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"

## Implemented in this repository

The first real-world readiness layer is now implemented. `src/realworld.py` provides strict manifest loading, camera-aware identity-balanced sampling, Market-style manifest generation for validation, and content-addressed gallery metadata. `train_upgraded.py` accepts either the original Market-1501 directory layout or a site manifest. `evaluate_real_world.py` reports overall and per-query-camera metrics, calibrates a review threshold against labeled query/gallery pairs, and writes a version record containing model, gallery, and metrics hashes.

The workflow was validated on the available demo subset with 795 training images, 200 queries, and 854 gallery images. It produced 62.0% top-1, 79.0% top-5, and 0.5037 mAP under same-camera filtering, plus a calibrated review threshold of 0.5897 at an observed validation false-positive rate of approximately 0.70%. These are benchmark validation results, not evidence of performance on a new site.
