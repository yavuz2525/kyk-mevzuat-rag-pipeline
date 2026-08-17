"""
Teable Knowledge Base Sync & Export Utility for RAG Pipeline.

This script syncs semantic markdown chunks and summaries to a Teable database
or exports them to structured JSON/CSV for downstream RAG vector stores.
"""

import os
import sys
import json
import csv
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_DIR = os.path.join(BASE_DIR, "output", "topics")
DATA_EXPORT_FILE = os.path.join(BASE_DIR, "teable_exported_data.json")

# Default metadata and descriptions for KYK Yurt Hizmetleri Yönetmeliği
TOPIC_METADATA = {
    "00_Kapak": {
        "subject": "Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği",
        "summary": "Bu belge, Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği'nin kapak sayfasıdır."
    },
    "01_Birinci_Bolum": {
        "subject": "Birinci Bölüm: Amaç, Kapsam, Dayanak ve Tanımlar",
        "summary": "Yönetmeliğin amacı (Madde 1), kapsamı (Madde 2), dayanağı (Madde 3) ve tanımları (Madde 4) içerir."
    },
    "02_Ikinci_Bolum": {
        "subject": "İkinci Bölüm: Yurt İhtiyacı, Barınma Şartları, Değerlendirme ve Yerleştirme İşlemleri",
        "summary": "Yurt ihtiyaç ve önceliklerinin tespiti (Madde 5), barınma şartları (Madde 6), başvuru ve belge araştırması (Madde 7), boş yatak tespiti (Madde 8), değerlendirme (Madde 9) ve yerleştirme ve kayıt işlemlerini (Madde 10) kapsar."
    },
    "03_Ucuncu_Bolum": {
        "subject": "Üçüncü Bölüm: Yurt Ücreti ve Güvence Bedeli",
        "summary": "Yurt ücreti (Madde 11), yurt ücretini ödemeyen öğrenciler (Madde 12), güvence bedeli (Madde 13) ve yurt ücreti alınmayacak öğrencilere (Madde 14) ilişkin düzenlemeleri içerir."
    },
    "04_Dorduncu_Bolum": {
        "subject": "Dördüncü Bölüm: Yurtların Açılması, Kapatılması ve Faaliyet Zamanları",
        "summary": "Yurtların açılması ve kapatılmasına (Madde 15) ve yurtların faaliyet zamanlarına (Madde 16) ilişkin düzenlemeleri içerir."
    },
    "05_Besinci_Bolum": {
        "subject": "Beşinci Bölüm: Barınma Süreleri ve İzinler",
        "summary": "Barınma süreleri (Madde 17), öğrenci izinleri (Madde 18), hastalık ve diğer sebeplerle yurttan geçici ayrılmalar (Madde 19) ve geçici barınmaya (Madde 20) ilişkin düzenlemeleri içerir."
    },
    "06_Altinci_Bolum": {
        "subject": "Altıncı Bölüm: Disiplin İşlemleri",
        "summary": "Disiplin cezaları (Madde 21-24), cezaların onay mercileri (Madde 25), cezanın tekrarı (Madde 26), ağırlaştırma/hafifletme (Madde 27), itiraz (Madde 28), disiplin kurulunun teşekkülü (Madde 29), çalışma usulleri (Madde 30), tebliğ (Madde 31), adli/disiplin işlemleri (Madde 32), yurtla ilişik kesme (Madde 33) ve yurt dışı disiplin işlemlerini (Madde 34) kapsar."
    },
    "07_Yedinci_Bolum": {
        "subject": "Yedinci Bölüm: Çeşitli ve Son Hükümler",
        "summary": "Misafirlerin ilişiğinin kesilmesi (Madde 35), gece hizmetleri (Madde 36), denetim (Madde 37), uluslararası öğrenciler (Madde 38), defter/dosyalar (Madde 39), başvuru sistemleri (Madde 40-41), usul ve esaslar (Madde 42), hizmet değerlendirme (Madde 43), atıflar (Madde 44), yürürlükten kaldırma (Madde 45), öğrenci konutu (Ek Madde 1), geçici maddeler (1-2) ve yürürlük/yürütme hükümlerini (Madde 46-47) içerir."
    }
}


def load_local_topics():
    """Load topics from output/topics directory with metadata."""
    records = []
    if not os.path.exists(TOPICS_DIR):
        print(f"Warning: {TOPICS_DIR} not found. Checking if export file exists...")
        if os.path.exists(DATA_EXPORT_FILE):
            with open(DATA_EXPORT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    topic_folders = sorted(os.listdir(TOPICS_DIR))
    auto_id = 1
    for folder in topic_folders:
        folder_path = os.path.join(TOPICS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        md_file = os.path.join(folder_path, f"{folder}.md")
        if not os.path.exists(md_file):
            continue

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Match metadata
        tag_key = folder
        meta = TOPIC_METADATA.get(tag_key, {
            "subject": folder.replace("_", " "),
            "summary": f"{folder} topic contents"
        })

        records.append({
            "id": auto_id,
            "subject": meta["subject"],
            "tag": tag_key,
            "summary": meta["summary"],
            "markdown_content": content
        })
        auto_id += 1

    return records


def export_to_json(records, output_path=DATA_EXPORT_FILE):
    """Export records to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {len(records)} records to {output_path}")


def export_to_csv(records, output_path=os.path.join(BASE_DIR, "teable_exported_data.csv")):
    """Export records to CSV file."""
    if not records:
        print("No records to export.")
        return
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject", "tag", "summary", "markdown_content"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Exported {len(records)} records to {output_path}")


def sync_to_teable_api(records, teable_url, table_id, api_token):
    """Sync records to Teable via REST API."""
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    url = f"{teable_url.rstrip('/')}/api/table/{table_id}/record"

    print(f"Syncing {len(records)} records to Teable ({url})...")
    payload = {
        "records": [
            {
                "fields": {
                    "Subject_String": r["subject"],
                    "Etiket": r["tag"],
                    "summary": r["summary"],
                    "markdown_content": r["markdown_content"]
                }
            }
            for r in records
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code in (200, 201):
            print("[OK] Successfully synced records to Teable!")
        else:
            print(f"[ERROR] Failed to sync to Teable API. Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Error during Teable API sync: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync or Export RAG knowledge base to/from Teable")
    parser.add_argument("--export-json", action="store_true", help="Export to teable_exported_data.json")
    parser.add_argument("--export-csv", action="store_true", help="Export to teable_exported_data.csv")
    parser.add_argument("--sync-api", action="store_true", help="Sync to Teable REST API")
    parser.add_argument("--teable-url", default=os.getenv("TEABLE_URL", "http://localhost:3000"), help="Teable base URL")
    parser.add_argument("--table-id", default=os.getenv("TEABLE_TABLE_ID", "tblbOZHZpc64B44Cuey"), help="Teable Table ID")
    parser.add_argument("--api-token", default=os.getenv("TEABLE_API_TOKEN", ""), help="Teable API Token")

    args = parser.parse_args()

    records = load_local_topics()
    print(f"Loaded {len(records)} topic records.")

    if args.export_json or (not args.export_csv and not args.sync_api):
        export_to_json(records)

    if args.export_csv:
        export_to_csv(records)

    if args.sync_api:
        if not args.api_token:
            print("❌ Please provide --api-token or set TEABLE_API_TOKEN environment variable.")
            sys.exit(1)
        sync_to_teable_api(records, args.teable_url, args.table_id, args.api_token)


if __name__ == "__main__":
    main()
