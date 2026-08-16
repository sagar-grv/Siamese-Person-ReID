# LinkedIn Post — Siamese Person Re-Identification

What do you do when a computer-vision project works in theory but breaks in practice?

You rebuild it from first principles.

I recently rebuilt and completed a **Siamese person re-identification system** that learns visual embeddings for cross-camera image retrieval. The original repository had missing data, broken imports, incomplete output handling, and no reliable browser deployment path. Instead of patching isolated errors, I redesigned the full workflow around reproducibility, evaluation, browser inference, and responsible real-world use.

The system now includes:

• A PyTorch training pipeline using a ResNet-18 backbone, BatchNorm neck, 256-dimensional L2-normalized embeddings, label-smoothed classification loss, and batch-hard triplet loss.

• Training and evaluation on the public Market-1501 benchmark, with disjoint identities between training and evaluation splits.

• ONNX export and a browser-native Vite application using ONNX Runtime Web, so image preprocessing, embedding extraction, and gallery ranking can run locally in the browser.

• A confidence-aware review gate. Instead of pretending that a similarity score is an identity decision, the interface labels results as **REVIEW CANDIDATE**, **LOW CONFIDENCE**, or **NO RELIABLE MATCH**.

• A calibrated review threshold of approximately **0.5897**, based on held-out validation data, together with content-addressed gallery versioning.

• An open-source self-hosted option using **FastAPI, FAISS, ONNX Runtime, Docker, Docker Compose, and Nginx**. This allows the project to run locally or on an independently managed server without depending on Vercel.

The verified demonstration returned 10 ranked candidates in the browser. The top score was **0.893**, above the calibrated review threshold, and the application correctly displayed **REVIEW CANDIDATE** while still requiring human confirmation.

On the full Market-1501 evaluation, the upgraded model achieved:

• **47.9% top-1 accuracy**
• **72.0% top-5 accuracy**
• **0.2773 mAP**

The most important lesson was not simply improving the model. It was building the surrounding system: dataset validation, camera-aware sampling, threshold calibration, gallery versioning, browser inference, deployment configuration, and documentation.

This project is intentionally a **research prototype for authorized visual retrieval**, not a surveillance or biometric-identification system. It must not be used for covert tracking or high-impact automated decisions. Similarity scores are review signals, not proof of identity, and human review remains mandatory.

The source code, self-hosting guide, API, Docker configuration, and training workflow are available here:

https://github.com/sagar-grv/Siamese-Person-ReID

Built with PyTorch, ONNX, ONNX Runtime Web, FastAPI, FAISS, Vite, and open-source tooling.

#MachineLearning #DeepLearning #ComputerVision #PersonReIdentification #SiameseNetwork #MetricLearning #PyTorch #ONNX #FAISS #FastAPI #OpenSource #MLOps #ResponsibleAI
