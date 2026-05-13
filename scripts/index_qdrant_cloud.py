#!/usr/bin/env python3
"""
Download Vietnamese legal retrieval data and index it into Qdrant Cloud.

This script intentionally uses only the Python standard library so it can run
on a clean local machine without requests/qdrant-client/OpenAI SDK.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class HttpResponseError(RuntimeError):
    pass


def http_request(
    method: str,
    url: str,
    headers: Dict[str, str] | None = None,
    payload: Dict | None = None,
    timeout: int = 60,
    ok_statuses: Tuple[int, ...] = (200,),
    retries: int = 3,
) -> Dict:
    body = None
    request_headers = headers or {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **request_headers}

    request = Request(url, data=body, headers=request_headers, method=method)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                status = response.status
                response_body = response.read()
            break
        except HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            if status < 500 and status != 429:
                break
            last_error = exc
        except (TimeoutError, URLError) as exc:
            last_error = exc
            status = 0
            response_body = b""

        if attempt < retries:
            sleep_s = min(2**attempt, 20)
            print(
                f"HTTP retry {attempt}/{retries} for {method} {url}: {last_error}. "
                f"Sleeping {sleep_s}s...",
                flush=True,
            )
            time.sleep(sleep_s)
    else:
        raise HttpResponseError(f"{method} {url} failed after {retries} retries: {last_error}")

    if status not in ok_statuses:
        message = response_body.decode("utf-8", errors="replace")[:500]
        raise HttpResponseError(f"{method} {url} returned {status}: {message}")

    if not response_body:
        return {}
    return json.loads(response_body.decode("utf-8"))


def download_source_to_cache(source_url: str, cache_file: Path) -> None:
    if cache_file.exists() and cache_file.stat().st_size > 0:
        print(f"Using cached source file: {cache_file}", flush=True)
        return

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    request = Request(
        source_url,
        headers={"User-Agent": "Vietnamese-Legal-RAG-Indexer/1.0"},
        method="GET",
    )

    for attempt in range(1, 4):
        try:
            print(f"Downloading source data to {cache_file}...", flush=True)
            with closing(urlopen(request, timeout=180)) as response, open(tmp_file, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_file.replace(cache_file)
            return
        except Exception as exc:
            if tmp_file.exists():
                tmp_file.unlink()
            if attempt == 3:
                raise
            sleep_s = min(2**attempt, 20)
            print(
                f"Download retry {attempt}/3 failed: {exc}. Sleeping {sleep_s}s...",
                flush=True,
            )
            time.sleep(sleep_s)


def iter_source_records(
    source_url: str, data_file: str | None, cache_file: Path | None = None
) -> Iterator[Dict]:
    if data_file:
        path = Path(data_file)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    if cache_file is None:
        cache_file = Path("data/cache/law_vi.jsonl.gz")
    download_source_to_cache(source_url, cache_file)
    with gzip.open(cache_file, "rt", encoding="utf-8") as handle:
        for line in handle:
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
        http_request(
            "DELETE",
            collection_url,
            headers=headers,
            timeout=60,
            ok_statuses=(200, 404),
        )

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

    http_request(
        "PUT",
        collection_url,
        headers=headers,
        payload=payload,
        timeout=60,
        ok_statuses=(200, 409),
    )


def embed_texts(
    base_url: str,
    api_key: str,
    model: str,
    texts: List[str],
) -> List[List[float]]:
    endpoint = f"{base_url.rstrip('/')}/embeddings"
    response = http_request(
        "POST",
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        payload={"model": model, "input": texts},
        timeout=120,
    )
    data = response["data"]
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
    http_request(
        "PUT",
        endpoint,
        headers={"api-key": qdrant_api_key, "Content-Type": "application/json"},
        payload={"points": points},
        timeout=120,
    )


def collection_info(qdrant_url: str, qdrant_api_key: str, collection: str) -> Dict:
    endpoint = f"{qdrant_url.rstrip('/')}/collections/{collection}"
    response = http_request(
        "GET",
        endpoint,
        headers={"api-key": qdrant_api_key},
        timeout=60,
    )
    return response.get("result", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="Index legal RAG data into Qdrant Cloud")
    parser.add_argument("--env-file", default="backend/.env")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--cache-file", default="data/cache/law_vi.jsonl.gz")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip source records before this zero-based index. Useful for manual resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the current Qdrant points_count. Assumes sequential numeric point IDs.",
    )
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

    print(f"Collection: {collection}", flush=True)
    print(f"Vector size: {vector_size}", flush=True)
    print(f"Distance: {distance}", flush=True)
    print(f"Limit: {args.limit}", flush=True)
    print("Creating collection if needed...", flush=True)
    create_or_recreate_collection(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection=collection,
        vector_size=vector_size,
        distance=distance,
        recreate=args.recreate,
    )

    start_index = max(args.start_index, 0)
    if args.resume:
        info = collection_info(qdrant_url, qdrant_api_key, collection)
        start_index = int(info.get("points_count") or 0)
        print(f"Resume enabled. Starting from Qdrant points_count={start_index}", flush=True)
    elif start_index:
        print(f"Starting from source index: {start_index}", flush=True)

    def prepared_records() -> Iterator[Tuple[int, str, Dict]]:
        for idx, record in enumerate(
            iter_source_records(args.source_url, args.data_file, Path(args.cache_file))
        ):
            if idx < start_index:
                continue
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
        print(f"Indexed batch {batch_index}: total={total}", flush=True)

    info = collection_info(qdrant_url, qdrant_api_key, collection)
    print("Done.", flush=True)
    print(f"Collection status: {info.get('status')}", flush=True)
    print(f"Points count: {info.get('points_count')}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
