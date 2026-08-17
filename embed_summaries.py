"""Embed KYK section summaries into a persistent ChromaDB vector store.

When Teable is fully configured, records are fetched from its API. Otherwise
the tracked ``teable_exported_data.json`` file is used as an offline fallback.
"""

import json
import os

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "teable_exported_data.json")

BASE_URL = os.getenv("TEABLE_URL", "http://127.0.0.1:3000").strip()
TABLE_ID = os.getenv("TEABLE_TABLE_ID", "").strip()
TOKEN = os.getenv("TEABLE_API_TOKEN", "").strip()


def _configured(value):
    return bool(value) and not value.startswith("your_")


def teable_is_configured():
    return _configured(TABLE_ID) and _configured(TOKEN)


def fetch_from_teable():
    """Fetch all records from Teable with pagination."""
    if not teable_is_configured():
        raise RuntimeError("Teable API token/table ID are not configured.")

    headers = {"Authorization": f"Bearer {TOKEN}"}
    all_records = []
    skip = 0
    take = 1000

    while True:
        url = f"{BASE_URL.rstrip('/')}/api/table/{TABLE_ID}/record"
        params = {"fieldKeyType": "name", "take": take, "skip": skip}
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        records = response.json().get("records", [])
        if not records:
            break

        all_records.extend(records)
        print(f"Fetched {len(records)} records (skip={skip}), total: {len(all_records)}")
        if len(records) < take:
            break
        skip += take

    return [
        {
            "id": record.get("id"),
            "subject": record.get("fields", {}).get("Subject_String", ""),
            "tag": record.get("fields", {}).get("Etiket", ""),
            "summary": record.get("fields", {}).get("summary", ""),
            "markdown_content": record.get("fields", {}).get("markdown_content", ""),
        }
        for record in all_records
    ]


def load_records():
    """Use Teable only when configured; otherwise load the local JSON dataset."""
    if teable_is_configured():
        try:
            print("[INFO] Attempting to fetch records from Teable API...")
            records = fetch_from_teable()
            if records:
                return records
            print("[WARN] Teable returned no records; falling back to local JSON dataset.")
        except Exception as exc:
            print(f"[WARN] Failed to fetch from Teable API ({exc}). Falling back to local JSON dataset.")

    if os.path.exists(LOCAL_DATA_PATH):
        print(f"[INFO] Loading records from local dataset: {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)

    raise FileNotFoundError("Neither Teable API nor local teable_exported_data.json is available.")


def embed_and_store():
    records = load_records()
    print(f"[INFO] Total records loaded: {len(records)}")

    texts = []
    metadatas = []
    ids = []
    for index, record in enumerate(records, 1):
        summary = str(record.get("summary", "")).strip()
        if not summary:
            continue

        record_id = str(record.get("id") or index)
        texts.append(summary)
        metadatas.append(
            {
                "record_id": record_id,
                "etiket": str(record.get("tag", "")),
                "subject": str(record.get("subject", "")),
                "summary_preview": summary[:200],
                "markdown_content": str(record.get("markdown_content", "")),
            }
        )
        ids.append(record_id)

    if not texts:
        raise RuntimeError("No valid summary texts found to embed.")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not _configured(openai_key):
        raise RuntimeError("Set OPENAI_API_KEY in .env before creating embeddings.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_key,
    )

    print("[INFO] Embedding summaries and persisting into ChromaDB...")
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=CHROMA_PATH,
        collection_name="kyk_summaries",
    )

    count = vector_store._collection.count()
    print(f"[OK] Stored {count} embeddings in ChromaDB at: {CHROMA_PATH}")


if __name__ == "__main__":
    embed_and_store()
