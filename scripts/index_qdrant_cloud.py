#!/usr/bin/env python3
"""
Download Vietnamese legal retrieval data and index it into Qdrant Cloud.

This script intentionally uses only the Python standard library plus `requests`
so it can run on a clean local machine without the qdrant-client/OpenAI SDK.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import requests


DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/anti-ai/ViNLI-Zalo-supervised/"
    "resolve/main/law_vi.jsonl.gz"
)


def load_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def require_env(values: Dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise RuntimeError(f"Missing required env value: {key}")
    return value


def iter_source_records(source_url: str, data_file: str | None) -> Iterator[Dict]:
    if data_file:
        path = Path(data_file)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    with requests.get(source_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8")
                if line.strip():
                    yield json.loads(line)


def normalize_record(record: Dict, idx: int) -> Tuple[str, str, Dict]:
    question = (
        record.get("question")
        or record.get("query")
        or record.get("instruction")
        or ""
    )
    context = (
        record.get("context")
        or record.get("positive")
        or record.get("answer")
        or record.get("output")
        or ""
    )

    question = str(question).replace("_", " ").strip()
    context = str(context).replace("_", " ").strip()

    if not question or not context:
        raise ValueError("missing question/context")

    payload = {
        "question": question,
        "content": context,
        "source": "anti-ai/ViNLI-Zalo-supervised/law_vi",
        "doc_id": idx,
    }
    return question, context, payload


def batched(items: Iterable[Tuple[int, str, Dict]], batch_size: int):
    batch: List[Tuple[int, str, Dict]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_or_recreate_collection(
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    vector_size: int,
    distance: str,
    recreate: bool,
) -> None:
    headers = {"api-key": qdrant_api_key, "Content-Type": "application/json"}
    collection_url = f"{qdrant_url.rstrip('/')}/collections/{collection}"

    if recreate:
        response = requests.delete(collection_url, headers=headers, timeout=60)
        if response.status_code not in (200, 404):
            response.raise_for_status()

    distance_map = {
        "COSINE": "Cosine",
        "DOT": "Dot",
        "EUCLID": "Euclid",
        "MANHATTAN": "Manhattan",
    }
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": distance_map.get(distance.upper(), "Cosine"),
        }
    }

    response = requests.put(collection_url, headers=headers, json=payload, timeout=60)
    if response.status_code not in (200, 409):
        response.raise_for_status()


def embed_texts(
    base_url: str,
    api_key: str,
    model: str,
    texts: List[str],
) -> List[List[float]]:
    endpoint = f"{base_url.rstrip('/')}/embeddings"
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": model, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()["data"]
    if data and "index" in data[0]:
        data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def upsert_points(
    qdrant_url: str,
    qdrant_api_key: str,
    collection: str,
    points: List[Dict],
) -> None:
    endpoint = f"{qdrant_url.rstrip('/')}/collections/{collection}/points?wait=true"
    response = requests.put(
        endpoint,
        headers={"api-key": qdrant_api_key, "Content-Type": "application/json"},
        json={"points": points},
        timeout=120,
    )
    response.raise_for_status()


def collection_info(qdrant_url: str, qdrant_api_key: str, collection: str) -> Dict:
    endpoint = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    response = requests.get(endpoint, headers={"api-key": qdrant_api_key}, timeout=60)
    response.raise_for_status()
    return response.json().get("result", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Index legal RAG data into Qdrant Cloud")
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    env = load_env(Path(args.env_file))
    qdrant_url = require_env(env, "QDRANT_URL")
    qdrant_api_key = require_env(env, "QDRANT_API_KEY")
    collection = args.collection or env.get("QDRANT_COLLECTION", "llm")
    vector_size = int(env.get("QDRANT_VECTOR_SIZE", "3072"))
    distance = env.get("QDRANT_DISTANCE", "COSINE")

    embedding_api_key = (
        env.get("EMBEDDING_API_KEY")
        or env.get("GEMINI_API_KEY")
        or env.get("GOOGLE_API_KEY")
    )
    if not embedding_api_key:
        raise RuntimeError("Missing EMBEDDING_API_KEY/GEMINI_API_KEY/GOOGLE_API_KEY")

    embedding_base_url = (
        env.get("EMBEDDING_BASE_URL")
        or env.get("GEMINI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    embedding_model = env.get("EMBEDDING_MODEL", "gemini-embedding-001")

    print(f"Collection: {collection}")
    print(f"Vector size: {vector_size}")
    print(f"Distance: {distance}")
    print(f"Limit: {args.limit}")
    print("Creating collection if needed...")
    create_or_recreate_collection(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection=collection,
        vector_size=vector_size,
        distance=distance,
        recreate=args.recreate,
    )

    def prepared_records() -> Iterator[Tuple[int, str, Dict]]:
        for idx, record in enumerate(iter_source_records(args.source_url, args.data_file)):
            if args.limit and idx >= args.limit:
                break
            try:
                question, context, payload = normalize_record(record, idx)
            except ValueError:
                continue
            text = f"{question} {context}"
            yield idx, text, payload

    total = 0
    for batch_index, batch in enumerate(batched(prepared_records(), args.batch_size), start=1):
        texts = [item[1] for item in batch]
        vectors = embed_texts(embedding_base_url, embedding_api_key, embedding_model, texts)
        points = [
            {
                "id": item[0],
                "vector": vector,
                "payload": item[2],
            }
            for item, vector in zip(batch, vectors)
        ]
        upsert_points(qdrant_url, qdrant_api_key, collection, points)
        total += len(points)
        print(f"Indexed batch {batch_index}: total={total}")

    info = collection_info(qdrant_url, qdrant_api_key, collection)
    print("Done.")
    print(f"Collection status: {info.get('status')}")
    print(f"Points count: {info.get('points_count')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
