"""
Embeds summary texts into ChromaDB vector store.
Can fetch data dynamically from Teable API or use the local dataset (teable_exported_data.json).
"""

import os
import json
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "teable_exported_data.json")

# Teable API Configuration
BASE_URL = os.getenv("TEABLE_URL", "http://127.0.0.1:3000")
TABLE_ID = os.getenv("TEABLE_TABLE_ID", "tblbOZHZpc64B44Cuey")
TOKEN = os.getenv("TEABLE_API_TOKEN", "")


def fetch_from_teable():
    """Fetch all records from Teable with pagination."""
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    all_records = []
    skip = 0
    take = 1000

    while True:
        url = f"{BASE_URL.rstrip('/')}/api/table/{TABLE_ID}/record"
        params = {"fieldKeyType": "name", "take": take, "skip": skip}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("records", [])

        if not records:
            break

        all_records.extend(records)
        print(f"Fetched {len(records)} records (skip={skip}), total: {len(all_records)}")

        if len(records) < take:
            break
        skip += take

    formatted = []
    for r in all_records:
        fields = r.get("fields", {})
        formatted.append({
            "id": r.get("id"),
            "subject": fields.get("Subject_String", ""),
            "tag": fields.get("Etiket", ""),
            "summary": fields.get("summary", ""),
            "markdown_content": fields.get("markdown_content", "")
        })
    return formatted


def load_records():
    """Try fetching from Teable API if configured, otherwise load local JSON dataset."""
    if TOKEN:
        try:
            print("[INFO] Attempting to fetch records from Teable API...")
            records = fetch_from_teable()
            if records:
                return records
        except Exception as e:
            print(f"[WARN] Failed to fetch from Teable API ({e}). Falling back to local JSON dataset.")

    if os.path.exists(LOCAL_DATA_PATH):
        print(f"[INFO] Loading records from local dataset: {LOCAL_DATA_PATH}")
        with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError("Neither Teable API nor local teable_exported_data.json found.")


def embed_and_store():
    records = load_records()
    print(f"[INFO] Total records loaded: {len(records)}")

    texts = []
    metadatas = []
    ids = []

    for r in records:
        summary = r.get("summary", "")
        if not summary or not summary.strip():
            continue

        texts.append(summary)
        metadatas.append({
            "record_id": str(r.get("id", "")),
            "etiket": r.get("tag", ""),
            "subject": r.get("subject", ""),
            "summary_preview": summary[:200],
            "markdown_content": r.get("markdown_content", ""),
        })
        ids.append(str(r.get("id", len(ids) + 1)))

    if not texts:
        print("[WARN] No valid summary texts found to embed.")
        return

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key.startswith("your_"):
        print("[ERROR] Please set OPENAI_API_KEY in your .env file.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
