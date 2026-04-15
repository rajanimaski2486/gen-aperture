"""
Stage 1 retriever for Search by Brief.

What this does
--------------
- Loads OpenAI CLIP locally
- Optionally loads a PCA model from a local .pkl file
- Embeds each Stage 0 lane embedding_query
- Calls the creativeImageSearchByEmbedding GraphQL service
- Returns a merged candidate pool of ids
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import clip
import numpy as np
import requests
import torch
from app.config import settings

DEFAULT_SEARCH_ENDPOINT = (
    "http://creative-image-similarity-search.sstk-ai-eng-prod.ct.shuttercloud.org/graphql"
)


# -----------------------------------------------------------------------------
# Local CLIP embedder
# -----------------------------------------------------------------------------

class LocalClipTextEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        download_root: str = "/tmp/clip",
        pca_model_path: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model, _ = clip.load(model_name, device=device, download_root=download_root)
        self.model.eval()

        self.pca_model = None
        if pca_model_path:
            self.pca_model = PortablePCAProjector.load(pca_model_path)

    def embed_texts(
        self,
        texts: list[str],
        pca: bool = False,
        normalize: bool = True,
        truncate: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        text_tensor = clip.tokenize(texts, truncate=truncate)

        with torch.no_grad():
            feat = self.model.encode_text(text_tensor.to(self.device))
            if normalize:
                feat = feat / feat.norm(dim=-1, keepdim=True)

        vectors = feat.detach().cpu().numpy()

        if pca:
            if self.pca_model is None:
                raise ValueError("PCA requested but no PCA model was loaded.")
            vectors = self.pca_model.transform(vectors)

        return vectors.tolist()


# -----------------------------------------------------------------------------
# Search service caller
# -----------------------------------------------------------------------------

def build_embedding_search_graphql(
    embedding: list[float],
    top_k: int = 500,
    collection_type: str = "APPROVED_V1",
) -> str:
    embedding_json = json.dumps([embedding])
    return (
        "query{\n"
        f"  creativeImageSearchByEmbedding (where: {{embeddings: {embedding_json}, collectionType: {collection_type}}}, top:{top_k}) "
        "{\n"
        "    modelName\n"
        "    similarImages {\n"
        "      entities {\n"
        "        classicId\n"
        "        score\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}"
    )


def call_embedding_search_service(
    embedding: list[float],
    endpoint: str = DEFAULT_SEARCH_ENDPOINT,
    top_k: int = 500,
    collection_type: str = "APPROVED_V1",
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    payload = {
        "query": build_embedding_search_graphql(
            embedding=embedding,
            top_k=top_k,
            collection_type=collection_type,
        )
    }
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def extract_entities_from_search_response(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    data = response_json.get("data", {})
    root = data.get("creativeImageSearchByEmbedding", {})
    similar_images = root.get("similarImages", [])

    # GraphQL can return similarImages either as:
    # - [{"entities": [...]}] (current service shape)
    # - {"entities": [...]}    (older helper expectation)
    if isinstance(similar_images, list):
        entities: list[dict[str, Any]] = []
        for item in similar_images:
            if isinstance(item, dict):
                bucket = item.get("entities", [])
                if isinstance(bucket, list):
                    entities.extend(bucket)
        return entities

    if isinstance(similar_images, dict):
        entities = similar_images.get("entities", [])
        return entities if isinstance(entities, list) else []

    return [] 


class PortablePCAProjector:
    """NumPy-only PCA projector loaded from exported .npz weights."""

    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: Optional[np.ndarray] = None,
        whiten: bool = False,
    ):
        self.components = components
        self.mean = mean
        self.explained_variance = explained_variance
        self.whiten = whiten

    @classmethod
    def load(cls, path: str) -> "PortablePCAProjector":
        if not path.endswith(".npz"):
            raise ValueError(
                f"Unsupported PCA file format: {path}. "
                "Use exported NumPy weights (.npz)."
            )

        weights = np.load(path)
        components = np.asarray(weights["components"], dtype=np.float32)
        mean = np.asarray(weights["mean"], dtype=np.float32)
        explained_variance = (
            np.asarray(weights["explained_variance"], dtype=np.float32)
            if "explained_variance" in weights
            else None
        )
        whiten = bool(int(weights["whiten"][0])) if "whiten" in weights else False
        return cls(
            components=components,
            mean=mean,
            explained_variance=explained_variance,
            whiten=whiten,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        transformed = (x - self.mean) @ self.components.T
        if self.whiten and self.explained_variance is not None:
            transformed = transformed / np.sqrt(self.explained_variance + 1e-12)
        return transformed


def merge_candidate_results(lane_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for lane_result in lane_results:
        lane_name = lane_result["lane_name"]
        lane_query = lane_result["lane_query"]
        entities = lane_result["entities"]

        for entity in entities:
            classic_id = str(entity["classicId"])
            score = entity.get("score")

            if classic_id not in merged:
                merged[classic_id] = {
                    "asset_id": classic_id,
                    "max_retrieval_score": score,
                    "origin_lane_names": [lane_name],
                    "origin_lane_queries": [lane_query],
                }
                continue

            if score is not None:
                existing = merged[classic_id].get("max_retrieval_score")
                if existing is None or score > existing:
                    merged[classic_id]["max_retrieval_score"] = score

            if lane_name not in merged[classic_id]["origin_lane_names"]:
                merged[classic_id]["origin_lane_names"].append(lane_name)

            if lane_query not in merged[classic_id]["origin_lane_queries"]:
                merged[classic_id]["origin_lane_queries"].append(lane_query)

    return sorted(
        merged.values(),
        key=lambda x: (x.get("max_retrieval_score") is not None, x.get("max_retrieval_score")),
        reverse=True,
    )


def _default_pca_path() -> str | None:
    # .../backend/app/services/searchbybrief/retriever.py -> repo root
    repo_root = Path(__file__).resolve().parents[4]
    npz_path = repo_root / "ipca_10m.npz"
    if npz_path.exists():
        return str(npz_path)
    pca_path = repo_root / "ipca_10m.pkl"
    return str(pca_path) if pca_path.exists() else None


def run_retriever_node(state: dict[str, Any]) -> dict[str, Any]:
    normalized_state = dict(state)
    search_params = normalized_state.get("search_params")
    if search_params is not None and hasattr(search_params, "model_dump"):
        normalized_state["search_params"] = search_params.model_dump()

    search_params_dict = normalized_state.get("search_params") or {}
    search_lanes = search_params_dict.get("search_lanes", [])
    if not search_lanes:
        return {
            **normalized_state,
            "candidate_pool": [],
            "lane_retrieval_results": [],
        }

    pca_model_path = settings.searchbybrief_retriever_pca_model_path or _default_pca_path()
    use_pca = settings.searchbybrief_retriever_use_pca
    top_k_per_lane = settings.searchbybrief_retriever_top_k_per_lane
    search_endpoint = settings.searchbybrief_retriever_endpoint or DEFAULT_SEARCH_ENDPOINT
    collection_type = settings.searchbybrief_retriever_collection_type
    clip_model_name = settings.searchbybrief_retriever_clip_model
    clip_device = settings.searchbybrief_retriever_clip_device
    clip_download_root = settings.searchbybrief_retriever_clip_download_root
    normalize = settings.searchbybrief_retriever_normalize_embeddings
    truncate = settings.searchbybrief_retriever_truncate_text
    timeout_seconds = settings.searchbybrief_retriever_timeout_seconds

    embedder = LocalClipTextEmbedder(
        model_name=clip_model_name,
        device=clip_device,
        download_root=clip_download_root,
        pca_model_path=pca_model_path,
    )
    lane_queries = [lane["embedding_query"] for lane in search_lanes]
    lane_embeddings = embedder.embed_texts(
        lane_queries,
        pca=use_pca,
        normalize=normalize,
        truncate=truncate,
    )

    lane_retrieval_results: list[dict[str, Any]] = []
    for lane, embedding in zip(search_lanes, lane_embeddings):
        response_json = call_embedding_search_service(
            embedding=embedding,
            endpoint=search_endpoint,
            top_k=top_k_per_lane,
            collection_type=collection_type,
            timeout_seconds=timeout_seconds,
        )
        entities = extract_entities_from_search_response(response_json)
        lane_retrieval_results.append(
            {
                "lane_name": lane["lane_name"],
                "lane_query": lane["embedding_query"],
                "embedding_dim": len(embedding),
                "entities": entities,
                "raw_response": response_json,
            }
        )

    candidate_pool = merge_candidate_results(lane_retrieval_results)
    return {
        **normalized_state,
        "candidate_pool": candidate_pool,
        "lane_retrieval_results": lane_retrieval_results,
    }
