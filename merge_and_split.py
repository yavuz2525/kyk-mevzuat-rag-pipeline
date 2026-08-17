"""Merge OCR page Markdown files, repair line breaks and split by regulation chapters.

The function validates all generated page inputs before replacing any prepared
outputs. If OCR page files are missing or incomplete, existing merged/topics
artifacts are left untouched.
"""

import os
import re
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MERGED_MD = os.path.join(OUTPUT_DIR, "merged_all_pages.md")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "topics")


def _page_number(dirname):
    suffix = dirname.removeprefix("page_")
    return int(suffix) if suffix.isdigit() else 10**9


def _load_page_markdown():
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}")

    page_dirs = [
        name
        for name in os.listdir(OUTPUT_DIR)
        if re.fullmatch(r"page_\d+", name)
        and os.path.isdir(os.path.join(OUTPUT_DIR, name))
    ]
    page_dirs.sort(key=_page_number)
    if not page_dirs:
        raise FileNotFoundError(
            "No generated output/page_N directories found. Run: python run_ocr_pages.py"
        )

    merged_parts = []
    missing = []
    for page_dir in page_dirs:
        md_path = os.path.join(OUTPUT_DIR, page_dir, f"{page_dir}.md")
        if not os.path.isfile(md_path):
            missing.append(md_path)
            continue
        with open(md_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            missing.append(md_path + " (empty)")
            continue
        merged_parts.append(content)

    if missing:
        preview = "\n  - ".join(missing[:10])
        raise RuntimeError(
            "OCR page output is incomplete; existing prepared outputs were not changed:\n"
            f"  - {preview}"
        )
    return merged_parts


def _repair_lines(full_text):
    lines = full_text.split("\n")
    fixed = []
    for line in lines:
        text = line.strip()
        if not text:
            if fixed and fixed[-1] != "":
                fixed.append("")
            continue
        if fixed and fixed[-1] != "":
            previous = fixed[-1]
            if (
                not previous.endswith((".", ":", ")", "—", "–", ",", '"'))
                and not re.match(r"^[A-ZÇĞİÖŞÜ0-9#(]", text)
                and not text.startswith("MADDE")
                and not text.startswith("##")
            ):
                fixed[-1] = previous + " " + text
                continue
        fixed.append(text)

    clean = []
    previous_blank = False
    for line in fixed:
        is_blank = line == ""
        if not is_blank or not previous_blank:
            clean.append(line)
        previous_blank = is_blank
    return "\n".join(clean).strip()


def _build_sections(merged_text):
    bolum_pattern = r"(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ)\s+BÖLÜM"
    matches = list(re.finditer(bolum_pattern, merged_text))
    if len(matches) < 7:
        raise RuntimeError(
            f"Expected 7 regulation chapter boundaries, found {len(matches)}. "
            "Existing prepared outputs were not changed."
        )

    sections = []
    title_match = re.search(r"^#\s+(.+)", merged_text, re.MULTILINE)
    if title_match and matches[0].start() > 0:
        cover = merged_text[: matches[0].start()].strip()
        if cover:
            sections.append(("00_Kapak", cover))

    translit = str.maketrans({"İ": "I", "Ü": "U", "Ö": "O", "Ş": "S", "Ç": "C", "Ğ": "G"})
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(merged_text)
        content = merged_text[match.start() : end].strip()
        if len(content) < 30:
            raise RuntimeError(
                f"Chapter {index + 1} is unexpectedly short. Existing prepared outputs were not changed."
            )
        ordinal = match.group(1).translate(translit).capitalize()
        sections.append((f"{index + 1:02d}_{ordinal}_Bolum", content))

    if len(sections) < 7:
        raise RuntimeError("No valid chapter sections produced; existing outputs were not changed.")
    return sections


def merge_and_split():
    print("[1/2] Validating and merging OCR pages...")
    merged_parts = _load_page_markdown()
    merged_text = _repair_lines("\n\n".join(merged_parts))
    if not merged_text:
        raise RuntimeError("Merged OCR text is empty; existing outputs were not changed.")

    print("[2/2] Validating regulation chapter boundaries...")
    sections = _build_sections(merged_text)

    # Only mutate prepared outputs after every validation above succeeds.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_merged = MERGED_MD + ".tmp"
    temp_topics = SPLIT_DIR + ".tmp"

    if os.path.exists(temp_topics):
        shutil.rmtree(temp_topics)
    os.makedirs(temp_topics, exist_ok=True)

    with open(temp_merged, "w", encoding="utf-8") as handle:
        handle.write(merged_text)

    for folder, content in sections:
        folder_path = os.path.join(temp_topics, folder)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f"{folder}.md"), "w", encoding="utf-8") as handle:
            handle.write(content)

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)
    os.replace(temp_topics, SPLIT_DIR)
    os.replace(temp_merged, MERGED_MD)

    print(f"  Merged: {len(merged_text)} chars -> {MERGED_MD}")
    for folder, content in sections:
        print(f"  {folder}/ ({content.count(chr(10)) + 1} lines, {len(content)} chars)")
    print(f"Done! {len(sections)} topic(s) saved to {SPLIT_DIR}")


if __name__ == "__main__":
    merge_and_split()
