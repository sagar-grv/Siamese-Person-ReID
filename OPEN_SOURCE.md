# Open-source deployment and licensing

This repository now provides two open-source execution paths. The existing browser demo performs ONNX inference and nearest-neighbor ranking entirely in the browser. The optional self-hosted service adds a FastAPI endpoint and FAISS vector index for larger galleries or integrations that should run on a local machine or an independently managed server.

> **Responsible-use boundary:** This project is a research and review aid for authorized datasets. It must not be used for surveillance, biometric identification, covert tracking, or high-impact automated decisions. A score is not an identity assertion. Every candidate requires a trained human reviewer and an explicit authorization process.

## License and data status

The project source code, container recipes, and service code are released under the MIT License in [`LICENSE`](LICENSE). Third-party dependencies retain their own licenses.

| Component | Role | License or access status |
|---|---|---|
| PyTorch and torchvision | Training and image transforms | Open-source software; retain upstream notices. |
| ONNX and ONNX Runtime | Model interchange and inference | ONNX Runtime is MIT-licensed; retain upstream notices [4]. |
| FAISS | Cosine-similarity vector search | MIT-licensed [2]. |
| FastAPI and Uvicorn | Self-hosted HTTP service | FastAPI is MIT-licensed; retain notices for all dependencies [3]. |
| Vite and ONNX Runtime Web | Browser application and client inference | Open-source dependencies; inspect the lockfile and retain upstream notices. |
| Market-1501 | Public benchmark used for reproducible evaluation | Public download source, but the official project page does not state a standalone OSI-approved data license. Download it directly from the official source and verify its terms before redistribution or commercial use [5]. |
| Site-specific images and galleries | Optional real-world deployment input | Must remain outside the repository unless the data owner has explicitly authorized redistribution. |

The model and gallery files checked into this repository are demonstration artifacts. Dataset and model terms are separate from the MIT license for the source code; users must audit the provenance and permitted use of any replacement data or weights. Torchreid remains an optional reference implementation for future model experiments, not a runtime dependency of this repository [1].

## Local open-source API

Install the API dependencies in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-server.txt
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

The service loads the existing ONNX encoder and gallery artifacts from `web/public`. It exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Reports model, gallery, and index readiness. |
| `GET /metadata` | Returns gallery version, embedding dimension, threshold, and review requirement. |
| `POST /search` | Accepts a multipart image upload and returns ranked review candidates. |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -F image=@web/public/data/demo-query.jpg \
  -F top_k=10
```

The response includes `review_state`, `review_threshold`, `gallery_version`, `human_review_required`, and ranked candidates. If `faiss-cpu` is unavailable, the service uses a NumPy cosine-search fallback, which keeps the API usable on platforms where FAISS wheels are not available.

## Docker Compose deployment

The repository includes a fully self-hostable two-container setup. It does not require Vercel or a proprietary runtime:

```bash
docker compose -f docker-compose.open-source.yml up --build
```

After startup, open `http://localhost:8080` for the browser demo, `http://localhost:8000/docs` for the local API documentation, and `http://localhost:8000/health` for a readiness check. The web container proxies `/api/` to the API container; the current browser demo remains browser-native and does not upload query images unless a future client integration explicitly opts into the API.

For an independently managed server, place the Compose stack behind HTTPS and restrict access to authorized operators. Set `REID_CORS_ORIGINS` to an explicit origin instead of `*`, keep private galleries outside public static directories, and rotate or remove any sample artifacts before production use.

## Replacing the benchmark gallery

The real-world workflow remains manifest-backed. Prepare an authorized CSV using [`examples/site_manifest.template.csv`](examples/site_manifest.template.csv), train with `train_upgraded.py --manifest`, evaluate with `evaluate_real_world.py`, and publish only the resulting model and gallery to a controlled deployment. Do not commit private images, personally identifying metadata, or raw camera footage.

The browser and API review gates consume the same `gallery_version.json` metadata. Recalibrate the threshold on held-out, site-specific validation data whenever the camera mix, illumination, detector, crop policy, or gallery changes.

## References

[1]: https://github.com/KaiyangZhou/deep-person-reid "Torchreid"
[2]: https://github.com/facebookresearch/faiss "FAISS"
[3]: https://github.com/fastapi/fastapi "FastAPI"
[4]: https://github.com/microsoft/onnxruntime "ONNX Runtime"
[5]: https://zheng-lab-anu.github.io/Project/project_reid.html "Official Market-1501 project page"
[6]: https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html "Market-1501 paper"

