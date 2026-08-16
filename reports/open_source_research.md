# Open-source implementation research

## Software stack

- Torchreid is an open-source PyTorch person-ReID library with training, evaluation, image/video ReID support, standard metrics, and ONNX export documentation: https://github.com/KaiyangZhou/deep-person-reid
- FAISS is a library for efficient similarity search and clustering of dense vectors; the official repository states it is MIT-licensed: https://github.com/facebookresearch/faiss
- FastAPI is an open-source Python API framework; the official repository provides a permissive MIT license: https://github.com/fastapi/fastapi
- ONNX Runtime is a cross-platform inference runtime; the official project is available under an MIT license: https://github.com/microsoft/onnxruntime

## Dataset caveat

The official Market-1501 project page describes 32,668 annotated bounding boxes across 1,501 identities and six cameras, with train, test, query, and ground-truth packages. It provides download links and a citation request, but the page does not state a standalone OSI-approved data license. Therefore the repository should remain open-source at the code level while users download Market-1501 directly from the official source and verify the dataset terms before redistribution or commercial use.

Official source: https://zheng-lab-anu.github.io/Project/project_reid.html
Paper: https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html

## Design decision

Add an optional self-hosted FastAPI service with FAISS-backed cosine search, while preserving the current browser-only ONNX demo. The service will load the repository's ONNX model and gallery JSON, accept an uploaded crop, return ranked candidates with a calibrated review state, and expose health and metadata endpoints. It will not perform identity decisions, surveillance, tracking, or automated action. Human review remains mandatory.

The code, container recipe, and service configuration can be distributed as open-source project files. Dataset images and any site-specific galleries remain external inputs and are not redistributed by the repository.
