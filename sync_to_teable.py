"""Teable knowledge-base sync and local export utility.

``--sync-api`` performs an idempotent upsert keyed by the ``Etiket`` field:
existing records are PATCHed and missing records are POSTed. Re-running the
command therefore does not create another copy of the same semantic section.
"""

import argparse
import csv
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPICS_DIR = os.path.join(BASE_DIR, "output", "topics")
DATA_EXPORT_FILE = os.path.join(BASE_DIR, "teable_exported_data.json")
CSV_EXPORT_FILE = os.path.join(BASE_DIR, "teable_exported_data.csv")

TOPIC_METADATA = {
    "00_Kapak": {
        "subject": "Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği",
        "summary": "Bu belge, Gençlik ve Spor Bakanlığı Yurt Hizmetleri Yönetmeliği'nin kapak sayfasıdır.",
    },
    "01_Birinci_Bolum": {
        "subject": "Birinci Bölüm: Amaç, Kapsam, Dayanak ve Tanımlar",
        "summary": "Yönetmeliğin amacı (Madde 1), kapsamı (Madde 2), dayanağı (Madde 3) ve tanımları (Madde 4) içerir.",
    },
    "02_Ikinci_Bolum": {
        "subject": "İkinci Bölüm: Yurt İhtiyacı, Barınma Şartları, Değerlendirme ve Yerleştirme İşlemleri",
        "summary": "Yurt ihtiyaç ve önceliklerinin tespiti (Madde 5), barınma şartları (Madde 6), başvuru ve belge araştırması (Madde 7), boş yatak tespiti (Madde 8), değerlendirme (Madde 9) ve yerleştirme ve kayıt işlemlerini (Madde 10) kapsar.",
    },
    "03_Ucuncu_Bolum": {
        "subject": "Üçüncü Bölüm: Yurt Ücreti ve Güvence Bedeli",
        "summary": "Yurt ücreti (Madde 11), yurt ücretini ödemeyen öğrenciler (Madde 12), güvence bedeli (Madde 13) ve yurt ücreti alınmayacak öğrencilere (Madde 14) ilişkin düzenlemeleri içerir.",
    },
    "04_Dorduncu_Bolum": {
        "subject": "Dördüncü Bölüm: Yurtların Açılması, Kapatılması ve Faaliyet Zamanları",
        "summary": "Yurtların açılması ve kapatılmasına (Madde 15) ve yurtların faaliyet zamanlarına (Madde 16) ilişkin düzenlemeleri içerir.",
    },
    "05_Besinci_Bolum": {
        "subject": "Beşinci Bölüm: Barınma Süreleri ve İzinler",
        "summary": "Barınma süreleri (Madde 17), öğrenci izinleri (Madde 18), hastalık ve diğer sebeplerle yurttan geçici ayrılmalar (Madde 19) ve geçici barınmaya (Madde 20) ilişkin düzenlemeleri içerir.",
    },
    "06_Altinci_Bolum": {
        "subject": "Altıncı Bölüm: Disiplin İşlemleri",
        "summary": "Disiplin cezaları (Madde 21-24), cezaların onay mercileri (Madde 25), cezanın tekrarı (Madde 26), ağırlaştırma/hafifletme (Madde 27), itiraz (Madde 28), disiplin kurulunun teşekkülü (Madde 29), çalışma usulleri (Madde 30), tebliğ (Madde 31), adli/disiplin işlemleri (Madde 32), yurtla ilişik kesme (Madde 33) ve yurt dışı disiplin işlemlerini (Madde 34) kapsar.",
    },
    "07_Yedinci_Bolum": {
        "subject": "Yedinci Bölüm: Çeşitli ve Son Hükümler",
        "summary": "Misafirlerin ilişiğinin kesilmesi (Madde 35), gece hizmetleri (Madde 36), denetim (Madde 37), uluslararası öğrenciler (Madde 38), defter/dosyalar (Madde 39), başvuru sistemleri (Madde 40-41), usul ve esaslar (Madde 42), hizmet değerlendirme (Madde 43), atıflar (Madde 44), yürürlükten kaldırma (Madde 45), öğrenci konutu (Ek Madde 1), geçici maddeler (1-2) ve yürürlük/yürütme hükümlerini (Madde 46-47) içerir.",
    },
}


def load_local_topics():
    records = []
    if not os.path.exists(TOPICS_DIR):
        if os.path.exists(DATA_EXPORT_FILE):
            print(f"[INFO] {TOPICS_DIR} not found; using {DATA_EXPORT_FILE}")
            with open(DATA_EXPORT_FILE, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return []

    auto_id = 1
    for folder in sorted(os.listdir(TOPICS_DIR)):
        folder_path = os.path.join(TOPICS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        md_file = os.path.join(folder_path, f"{folder}.md")
        if not os.path.exists(md_file):
            continue

        with open(md_file, "r", encoding="utf-8") as handle:
            content = handle.read().strip()

        meta = TOPIC_METADATA.get(
            folder,
            {"subject": folder.replace("_", " "), "summary": f"{folder} topic contents"},
        )
        records.append(
            {
                "id": auto_id,
                "subject": meta["subject"],
                "tag": folder,
                "summary": meta["summary"],
                "markdown_content": content,
            }
        )
        auto_id += 1

    return records


def export_to_json(records, output_path=DATA_EXPORT_FILE):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {len(records)} records to {output_path}")


def export_to_csv(records, output_path=CSV_EXPORT_FILE):
    if not records:
        print("[WARN] No records to export.")
        return
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "subject", "tag", "summary", "markdown_content"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Exported {len(records)} records to {output_path}")


def _headers(api_token):
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


def _record_fields(record):
    return {
        "Subject_String": record["subject"],
        "Etiket": record["tag"],
        "summary": record["summary"],
        "markdown_content": record["markdown_content"],
    }


def fetch_teable_records(teable_url, table_id, api_token):
    """Fetch all Teable records using name-based field keys and pagination."""
    url = f"{teable_url.rstrip('/')}/api/table/{table_id}/record"
    records = []
    skip = 0
    take = 1000

    while True:
        response = requests.get(
            url,
            headers=_headers(api_token),
            params={"fieldKeyType": "name", "take": take, "skip": skip},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json().get("records", [])
        records.extend(batch)
        if len(batch) < take:
            break
        skip += take

    return records


def sync_to_teable_api(records, teable_url, table_id, api_token):
    """Upsert local topic records to Teable by the unique ``Etiket`` value."""
    base_url = f"{teable_url.rstrip('/')}/api/table/{table_id}/record"
    existing = fetch_teable_records(teable_url, table_id, api_token)

    by_tag = {}
    duplicate_tags = set()
    for remote in existing:
        tag = remote.get("fields", {}).get("Etiket")
        if not tag:
            continue
        if tag in by_tag:
            duplicate_tags.add(tag)
            continue
        by_tag[tag] = remote

    if duplicate_tags:
        print(
            "[WARN] Existing duplicate Etiket values detected in Teable: "
            + ", ".join(sorted(duplicate_tags))
            + ". Sync updates one canonical record per tag and does not delete data automatically."
        )

    created = 0
    updated = 0
    create_payload = []

    for record in records:
        tag = record["tag"]
        fields = _record_fields(record)
        remote = by_tag.get(tag)
        if remote:
            record_id = remote.get("id")
            if not record_id:
                raise RuntimeError(f"Existing Teable record for {tag!r} has no id.")
            response = requests.patch(
                f"{base_url}/{record_id}",
                headers=_headers(api_token),
                json={
                    "fieldKeyType": "name",
                    "record": {"fields": fields},
                },
                timeout=30,
            )
            response.raise_for_status()
            updated += 1
        else:
            create_payload.append({"fields": fields})

    if create_payload:
        response = requests.post(
            base_url,
            headers=_headers(api_token),
            json={"fieldKeyType": "name", "records": create_payload},
            timeout=30,
        )
        response.raise_for_status()
        created = len(create_payload)

    print(f"[OK] Teable sync complete: {updated} updated, {created} created.")


def main():
    parser = argparse.ArgumentParser(description="Sync/export the KYK RAG knowledge base")
    parser.add_argument("--export-json", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--sync-api", action="store_true")
    parser.add_argument(
        "--teable-url", default=os.getenv("TEABLE_URL", "http://localhost:3000")
    )
    parser.add_argument("--table-id", default=os.getenv("TEABLE_TABLE_ID", ""))
    parser.add_argument("--api-token", default=os.getenv("TEABLE_API_TOKEN", ""))
    args = parser.parse_args()

    records = load_local_topics()
    if not records:
        print("[ERROR] No topic records found. Run merge_and_split.py or restore the JSON dataset.")
        sys.exit(1)
    print(f"Loaded {len(records)} topic records.")

    if args.export_json or (not args.export_csv and not args.sync_api):
        export_to_json(records)
    if args.export_csv:
        export_to_csv(records)
    if args.sync_api:
        if not args.api_token or args.api_token.startswith("your_"):
            print("[ERROR] Set TEABLE_API_TOKEN or pass --api-token.")
            sys.exit(1)
        if not args.table_id or args.table_id.startswith("your_"):
            print("[ERROR] Set TEABLE_TABLE_ID or pass --table-id.")
            sys.exit(1)
        sync_to_teable_api(records, args.teable_url, args.table_id, args.api_token)


if __name__ == "__main__":
    main()
