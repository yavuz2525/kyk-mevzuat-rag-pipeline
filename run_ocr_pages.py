"""DeepSeek-OCR pipeline for the bundled KYK regulation PDF.

Each PDF page is rendered, OCR'd and written to ``output/page_N`` with
Markdown, HTML, the original page image, a boxed preview and extracted figures.
Generated page directories are intentionally git-ignored.
"""

import ast
import io
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_PATH, "7.5.38510.pdf")
OUTPUT_BASE = os.path.join(BASE_PATH, "output")
TEMP_PAGES_DIR = os.path.join(OUTPUT_BASE, "temp_pages")

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DPI = 144


def get_page_count(pdf_path):
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def pdf_page_to_image(pdf_path, page_num, dpi=DPI):
    """Extract one zero-indexed PDF page as an RGB PIL image."""
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


def re_match(text):
    """Parse DeepSeek grounding ref/det tokens."""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    image_matches = [match[0] for match in matches if match[1] == "image"]
    other_matches = [match[0] for match in matches if match[1] != "image"]
    return matches, image_matches, other_matches


def extract_coords(ref_text):
    """Safely parse a ref tuple into ``(label, boxes)``."""
    try:
        raw_points = ast.literal_eval(ref_text[2])
    except (ValueError, SyntaxError):
        return None

    if not isinstance(raw_points, (list, tuple)):
        return None

    boxes = []
    for points in raw_points:
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in points)
        except (TypeError, ValueError):
            continue
        boxes.append([x1, y1, x2, y2])

    return (ref_text[1], boxes) if boxes else None


def _scale_box(points, img_w, img_h):
    values = [max(0.0, min(999.0, value)) for value in points]
    x1, y1, x2, y2 = values
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (
        int(x1 / 999 * img_w),
        int(y1 / 999 * img_h),
        int(x2 / 999 * img_w),
        int(y2 / 999 * img_h),
    )


def draw_boxes_and_extract(image, refs, page_dir, page_num):
    """Draw grounding boxes and save extracted figures.

    Returns ``(boxed_path, image_links)`` ordered like image ref tokens, so
    generated Markdown references always match the extracted files.
    """
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    images_dir = os.path.join(page_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    image_links = []
    image_ref_index = 0

    for ref in refs:
        result = extract_coords(ref)
        if not result:
            continue

        label_type, points_list = result
        color = tuple(
            int(v)
            for v in (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255),
            )
        )
        color_a = color + (20,)

        first_image_name = None
        for point_index, points in enumerate(points_list):
            x1, y1, x2, y2 = _scale_box(points, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue

            if label_type == "image":
                image_name = (
                    f"{image_ref_index}.jpg"
                    if len(points_list) == 1
                    else f"{image_ref_index}_{point_index}.jpg"
                )
                image.crop((x1, y1, x2, y2)).save(
                    os.path.join(images_dir, image_name), quality=95
                )
                if first_image_name is None:
                    first_image_name = image_name

            width = 4 if label_type == "title" else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            overlay_draw.rectangle(
                [x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1
            )

            tx, ty = x1, max(0, y1 - 18)
            bbox = draw.textbbox((0, 0), label_type, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([tx, ty, tx + tw, ty + th], fill=(255, 255, 255, 220))
            draw.text((tx, ty), label_type, font=font, fill=color)

        if label_type == "image":
            image_links.append(first_image_name)
            image_ref_index += 1

    img_draw.paste(overlay, (0, 0), overlay)
    boxed_path = os.path.join(page_dir, f"page_{page_num}_boxed.png")
    img_draw.save(boxed_path)
    return boxed_path, image_links


def md_to_html(md_text):
    """Convert the small Markdown subset emitted by the OCR model to HTML."""
    import html as html_mod

    paragraphs = md_text.split("\n\n")
    parts = [
        "<!DOCTYPE html>",
        '<html lang="tr">',
        "<head>",
        '<meta charset="UTF-8">',
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:800px;margin:40px auto;padding:20px;line-height:1.6;color:#333}",
        "h1{font-size:1.6em;border-bottom:2px solid #eee;padding-bottom:8px}",
        "h2{font-size:1.3em;border-bottom:1px solid #eee;padding-bottom:6px}",
        "h3{font-size:1.1em}",
        "table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f5f5f5}",
        "img{max-width:100%;height:auto}",
        "code{background:#f4f4f4;padding:2px 6px;border-radius:3px}pre{background:#f4f4f4;padding:16px;border-radius:6px;overflow-x:auto}",
        "</style></head><body>",
    ]

    in_table = False
    for para in paragraphs:
        if not para.strip():
            continue
        lines = para.split("\n")
        if lines[0].startswith("|") and lines[0].endswith("|"):
            if not in_table:
                parts.append("<table>")
                in_table = True
            for line in lines:
                if re.match(r"^\|[\s\-:]+\|", line):
                    continue
                cells = line.split("|")[1:-1]
                parts.append(
                    "<tr>"
                    + "".join(
                        f"<td>{html_mod.escape(cell.strip())}</td>" for cell in cells
                    )
                    + "</tr>"
                )
            continue

        if in_table:
            parts.append("</table>")
            in_table = False

        if para.startswith("### "):
            parts.append(f"<h3>{html_mod.escape(para[4:])}</h3>")
        elif para.startswith("## "):
            parts.append(f"<h2>{html_mod.escape(para[3:])}</h2>")
        elif para.startswith("# "):
            parts.append(f"<h1>{html_mod.escape(para[2:])}</h1>")
        else:
            content = html_mod.escape(para)
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
            content = re.sub(
                r"!\[(.*?)\]\((.+?)\)", r'<img src="\2" alt="\1">', content
            )
            content = re.sub(r"\[(.*?)\]\((.+?)\)", r'<a href="\2">\1</a>', content)
            parts.append(f"<p>{content}</p>")

    if in_table:
        parts.append("</table>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def clean_output(content, image_matches, other_matches, image_links):
    """Remove grounding tokens and replace image refs with valid local links."""
    for index, match in enumerate(image_matches):
        image_name = image_links[index] if index < len(image_links) else None
        replacement = f"![](images/{image_name})\n" if image_name else ""
        content = content.replace(match, replacement, 1)

    for match in other_matches:
        content = content.replace(match, "", 1)

    content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(
        r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>",
        "",
        content,
        flags=re.DOTALL,
    )
    return content.strip()


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Source PDF not found: {PDF_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DeepSeek-OCR 4-bit pipeline requires a CUDA-capable NVIDIA GPU."
        )

    total_pages = get_page_count(PDF_PATH)
    max_pages = int(os.getenv("OCR_MAX_PAGES", "0") or 0)
    page_count = min(total_pages, max_pages) if max_pages > 0 else total_pages

    print(f"[1/3] Rendering {page_count} PDF page(s)...")
    os.makedirs(TEMP_PAGES_DIR, exist_ok=True)
    images = []
    for page_index in range(page_count):
        img = pdf_page_to_image(PDF_PATH, page_index, DPI)
        images.append(img)
        img.save(os.path.join(TEMP_PAGES_DIR, f"page_{page_index + 1}.png"))
        print(f"  Page {page_index + 1}: {img.size[0]}x{img.size[1]}")

    print("\n[2/3] Loading DeepSeek-OCR model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="eager",
    )
    print(f"  Model loaded. GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print(f"\n[3/3] Running OCR on {page_count} page(s)...")
    for page_index, image in enumerate(images):
        page_num = page_index + 1
        page_dir = os.path.join(OUTPUT_BASE, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)
        img_path = os.path.join(TEMP_PAGES_DIR, f"page_{page_num}.png")

        print(f"  Processing page {page_num}...")
        output_text = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=img_path,
            output_path=page_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            eval_mode=True,
            test_compress=False,
            save_results=False,
        )
        if not output_text:
            print(f"    WARNING: no OCR output for page {page_num}")
            continue

        refs, image_matches, other_matches = re_match(output_text)
        boxed_path, image_links = draw_boxes_and_extract(
            image.copy(), refs, page_dir, page_num
        )
        md_content = clean_output(
            output_text, image_matches, other_matches, image_links
        )

        md_path = os.path.join(page_dir, f"page_{page_num}.md")
        html_path = os.path.join(page_dir, f"page_{page_num}.html")
        original_path = os.path.join(page_dir, f"page_{page_num}_original.png")

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(md_content)
        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(md_to_html(md_content))
        image.save(original_path)

        print(
            f"    refs: {len(refs)} | extracted image refs: "
            f"{sum(1 for item in image_links if item)}"
        )
        print(f"    -> {md_path}")
        print(f"    -> {html_path}")
        print(f"    -> {boxed_path}")
        print(f"    -> {original_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
