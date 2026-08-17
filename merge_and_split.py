"""
Merge all page markdowns into one, then split by BOLUM chapters.
"""
import os
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MERGED_MD = os.path.join(OUTPUT_DIR, "merged_all_pages.md")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "topics")


def merge_and_split():
    # Clean up old topic dirs
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # Step 1: Merge all pages
    print("[1/2] Merging all pages...")
    merged_parts = []
    
    # Auto-detect number of page directories
    page_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("page_") and os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    page_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999)

    for pdir in page_dirs:
        md_path = os.path.join(OUTPUT_DIR, pdir, f"{pdir}.md")
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                merged_parts.append(content)

    full_text = "\n\n".join(merged_parts)

    # Fix broken lines across page boundaries: join lines that don't end with punctuation
    lines = full_text.split("\n")
    fixed = []
    for line in lines:
        s = line.strip()
        if not s:
            if fixed and fixed[-1] != "":
                fixed.append("")
            continue
        if fixed and fixed[-1] != "":
            prev = fixed[-1]
            if not prev.endswith((".", ":", ")", "—", "–", ",")) and not prev.endswith('"'):
                if not re.match(r'^[A-Z0-9#(]', s) and not s.startswith("MADDE") and not s.startswith("##"):
                    fixed[-1] = prev + " " + s
                    continue
        fixed.append(s)

    # Remove excessive blank lines
    clean = []
    prev_blank = False
    for line in fixed:
        if line == "":
            if not prev_blank:
                clean.append(line)
            prev_blank = True
        else:
            clean.append(line)
            prev_blank = False

    merged_text = "\n".join(clean).strip()
    with open(MERGED_MD, "w", encoding="utf-8") as f:
        f.write(merged_text)
    print(f"  Merged -> {len(merged_text)} chars saved to {MERGED_MD}")

    # Step 2: Split by BOLUM chapters
    print("\n[2/2] Splitting by BOLUM chapters...")

    chapter_map = [
        ("BIRINCI BOLUM",  "Amaç, Kapsam, Dayanak ve Tanımlar (MADDE 1-4)"),
        ("IKINCI BOLUM",   "Yurt İhtiyacı, Barınma Şartları, Değerlendirme ve Yerleştirme (MADDE 5-10)"),
        ("UCUNCU BOLUM",   "Yurt Ücreti ve Güvence Bedeli (MADDE 11-14)"),
        ("DORDUNCU BOLUM", "Yurtların Açılması, Kapatılması ve Faaliyet Zamanları (MADDE 15-16)"),
        ("BESINCI BOLUM",  "Barınma Süreleri ve İzinler (MADDE 17-20)"),
        ("ALTINCI BOLUM",  "Disiplin İşlemleri (MADDE 21-34)"),
        ("YEDINCI BOLUM",  "Çeşitli ve Son Hükümler (MADDE 35-47)"),
    ]

    bolum_pattern = r'(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ)\s+BÖLÜM'
    bolum_matches = list(re.finditer(bolum_pattern, merged_text))

    print(f"  Found {len(bolum_matches)} chapter boundaries")

    title_match = re.search(r"^#\s+(.+)", merged_text, re.MULTILINE)
    raw_sections = []

    if title_match:
        raw_sections.append({
            "start": 0,
            "end": bolum_matches[0].start() if bolum_matches else title_match.end(),
            "folder": "00_Kapak",
            "desc": "Yonetmelik Basligi",
        })

    for i, match in enumerate(bolum_matches):
        start = match.start()
        if i + 1 < len(bolum_matches):
            end = bolum_matches[i + 1].start()
        else:
            end = len(merged_text)

        bolum_ordinal = match.group(1)
        folder_key = bolum_ordinal.replace("İ", "I").replace("Ü", "U").replace("Ö", "O").replace("Ş", "S").replace("Ç", "C").replace("Ğ", "G")
        folder_key = folder_key.replace("ı", "i").replace("ü", "u").replace("ö", "o").replace("ş", "s").replace("ç", "c").replace("ğ", "g")

        desc = ""
        for key, d in chapter_map:
            if key in folder_key.upper():
                desc = d
                break

        folder_name = f"{i+1:02d}_{folder_key.capitalize()}_Bolum"

        content = merged_text[start:end].strip()

        raw_sections.append({
            "start": start,
            "end": end,
            "folder": folder_name,
            "desc": desc,
            "content": content,
        })

    seen_starts = set()
    final_sections = []
    for s in raw_sections:
        content = s.get("content", merged_text[s["start"]:s["end"]]).strip()
        if len(content) < 30:
            continue
        if s["start"] in seen_starts:
            continue
        seen_starts.add(s["start"])
        s["content"] = content
        final_sections.append(s)

    for s in final_sections:
        folder_path = os.path.join(SPLIT_DIR, s["folder"])
        os.makedirs(folder_path, exist_ok=True)

        file_name = s["folder"] + ".md"
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(s["content"])

        lines = s["content"].count("\n") + 1
        print(f"  {s['folder']}/  ({lines} lines, {len(s['content'])} chars)")

    print(f"\nDone! {len(final_sections)} topics saved to {SPLIT_DIR}")


if __name__ == "__main__":
    merge_and_split()
