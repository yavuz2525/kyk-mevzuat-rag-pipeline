"""
CLI tool to query ChromaDB embeddings and inspect similarity scores.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")


def query_similar(query_text: str, k: int = 4):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="kyk_summaries",
    )

    results = vector_store.similarity_search_with_score(query_text, k=k)

    print(f"\n[QUERY]: \"{query_text}\"")
    print(f"Bulunan sonuc sayisi: {len(results)}\n")
    print("=" * 70)

    for i, (doc, score) in enumerate(results, 1):
        tag = doc.metadata.get("etiket", "N/A")
        subj = doc.metadata.get("subject", "N/A")
        print(f"#{i} | Benzerlik Skoru: {score:.4f}")
        print(f"   | Bolum Etiketi: {tag}")
        print(f"   | Konu:          {subj}")
        print(f"   | Ozet:          {doc.page_content[:200]}...")
        print("-" * 70)

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_similar(" ".join(sys.argv[1:]))
    else:
        print("Kullanim: python query_embeddings.py \"sorunuz buraya\"")
