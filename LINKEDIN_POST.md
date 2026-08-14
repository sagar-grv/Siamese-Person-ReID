# LinkedIn post draft

## From a broken repository to a browser-native Siamese ReID system

A few days ago, I opened a person re-identification repository expecting to fix a small issue.

Instead, I found a project that could not run end to end: the annotation file was present, but the referenced image directory was missing; trained model artifacts were not available; and the original application architecture was not a natural fit for a lightweight public deployment.

That became a much more interesting engineering problem:

**Could I rebuild the project from the metric-learning idea, train it on a public person ReID dataset, evaluate it on identities the model had never seen, and make the result run directly in a browser?**

The answer became a new Siamese ReID prototype called **Trace / Lab**.

The core idea is simple but powerful. Two person images pass through the same encoder with shared weights. A contrastive loss pulls images of the same person closer in embedding space and pushes images of different people apart. Instead of predicting one of a fixed list of identities, the model learns a visual similarity function that can be used for retrieval.

I used a public Market-1501 subset and deliberately kept training identities separate from evaluation identities. That distinction matters: a ReID model should be tested on its ability to generalize to new people, not only memorize the people it saw during training.

The rebuilt pipeline includes:

• A compact convolutional Siamese encoder
• 128-dimensional L2-normalized embeddings
• On-the-fly positive and negative pair generation
• Contrastive loss with a margin of 1.0
• Deterministic training with seed 42
• Top-1, Top-5, and mean average precision evaluation
• Self-contained ONNX export
• Browser-side inference with ONNX Runtime Web
• A Vite frontend deployable to Vercel

On the reproducible subset experiment, the system achieved:

→ 58.5% Top-1 retrieval accuracy
→ 82.5% Top-5 retrieval accuracy
→ 0.3296 mean average precision

These are not state-of-the-art claims. They are honest prototype measurements from a small subset, and that distinction is important. The goal was to build a transparent, working system with an evaluation protocol—not to hide the limitations behind a polished interface.

The most satisfying part was seeing the entire loop work:

Upload a person crop → encode it locally in the browser → compare it with a precomputed gallery → render the nearest cross-camera views.

No Python server. No API key. No image upload endpoint. The model and gallery index are static assets, and inference runs locally in the user’s browser.

This project also reminded me that machine learning engineering is rarely just about writing a model class. The difficult parts were equally practical:

• Finding a usable public dataset source
• Making the train/evaluation identity split explicit
• Keeping Python and browser preprocessing identical
• Debugging ONNX external-data loading
• Understanding how Vercel Root Directory changes build commands
• Communicating what the model can and cannot claim

The next improvements would be full-dataset training, official Market-1501 camera-aware evaluation, hard-negative mining, stronger backbones, multiple random seeds, and more rigorous privacy and governance controls.

For now, this is a small but complete example of turning a fragile research project into a reproducible ML product surface.

The code, technical documentation, metrics, and deployment configuration are available here:

https://github.com/sagar-grv/Siamese-Person-ReID

What would you improve first: the model architecture, the evaluation protocol, or the production system around it?

#MachineLearning #DeepLearning #ComputerVision #PersonReIdentification #SiameseNetwork #MetricLearning #MLOps #ONNX #Vercel #Python
