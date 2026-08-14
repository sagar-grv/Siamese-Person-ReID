# Rebuild research notes

## Paper2Code reference

Source: https://github.com/going-doer/Paper2Code

The repository describes a three-stage pipeline: planning, analysis, and code generation, handled by specialized agents. Its runnable workflow is organized under `scripts/`, with outputs separated into planning, analyzing, and coding artifacts plus a final generated repository. For this rebuild, the practical adaptation is to keep a written implementation plan, an experiment/evaluation artifact, and a runnable final repository rather than copying the LLM orchestration itself.

## Market-1501 dataset

Source: https://zheng-lab-anu.github.io/Project/project_reid.html

The official dataset page describes Market-1501 as a six-camera person re-identification dataset with 1,501 identities and 32,668 annotated bounding boxes. It lists 12,936 training images, 19,732 bounding-box test images, and 3,368 query images, plus ground-truth query annotations. It provides Google Drive, Baidu, and direct ZIP download links. The dataset is appropriate for cross-camera retrieval evaluation, but the full archive is large; the rebuild should support a documented, deterministic subset for local/CI training while preserving identity-disjoint evaluation.

## Implementation direction

Use the original Siamese-network idea as the core: two images pass through shared weights to produce embeddings, and a contrastive objective pulls same-identity pairs together while pushing different-identity pairs apart. For the real-world demo, expose query-image upload, top-k gallery retrieval, similarity scores, and a clear disclaimer that this is a research/demo model rather than a biometric identity service. Separate training/evaluation artifacts from the web deployment; Vercel should host a lightweight frontend and inference API only if the final model artifact and runtime fit serverless limits.
