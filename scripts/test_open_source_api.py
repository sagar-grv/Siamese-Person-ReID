import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app import app


IMAGE = ROOT / "web/public/data/demo-query.jpg"


def main() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text
    health_payload = health.json()
    assert health_payload["model_loaded"] is True, health_payload
    assert health_payload["gallery_images"] == 854, health_payload
    assert health_payload["index_backend"] == "faiss", health_payload

    metadata = client.get("/metadata")
    assert metadata.status_code == 200, metadata.text
    metadata_payload = metadata.json()
    assert metadata_payload["embedding_dim"] == 256, metadata_payload
    assert metadata_payload["human_review_required"] is True, metadata_payload
    assert metadata_payload["review_threshold"] == 0.5896926021575923, metadata_payload

    with IMAGE.open("rb") as handle:
        response = client.post(
            "/search",
            files={"image": (IMAGE.name, handle, "image/jpeg")},
            params={"top_k": 10},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["candidates"]) == 10, payload
    assert payload["review_state"] in {"REVIEW CANDIDATE", "NO RELIABLE MATCH"}, payload
    assert payload["human_review_required"] is True, payload
    assert payload["gallery_version"] == "df0ef1d10ccb-92f18a0447f7", payload

    print("health", health_payload)
    print("metadata", metadata_payload)
    print("review_state", payload["review_state"])
    print("top_score", payload["candidates"][0]["score"])


if __name__ == "__main__":
    main()
