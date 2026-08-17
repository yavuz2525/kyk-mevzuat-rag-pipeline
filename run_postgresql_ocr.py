"""
DeepSeek OCR - PostgreSQL PDF sayfa 21 ve 23 işleme
Çıktı: postgresql_ocr_output/ dizini altında her sayfa için ayrı klasör
  - .md (markdown)
  - .html
  - _boxed.png (kutucuklu görsel)
  - _original.png
  - images/ (içindeki figürler)
"""
import os, sys, re, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# === PATHS ===
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_PATH, "postgresql_tutorial.pdf")
OUTPUT_BASE = os.path.join(BASE_PATH, "postgresql_ocr_output")
PAGES_TO_PROCESS = [20, 22]  # 0-indexed: page 21 and 23

# === CONFIG ===
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DPI = 144


def pdf_page_to_image(pdf_path, page_num, dpi=144):
    """Extract a single PDF page as PIL Image."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    doc.close()
    return img


def re_match(text):
    """Parse <|ref|>...<|/ref|><|det|>...<|/det|> patterns."""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    mathes_image = []
    mathes_other = []
    for match in matches:
        if "<|ref|>image<|/ref|>" in match[0]:
            mathes_image.append(match[0])
        else:
            mathes_other.append(match[0])
    return matches, mathes_image, mathes_other


def extract_coords(ref_text):
    """Extract (label_type, [[x1,y1,x2,y2], ...]) from a ref match tuple."""
    try:
        return (ref_text[1], eval(ref_text[2]))
    except Exception:
        return None


def draw_boxes_and_extract(image, refs, page_dir, real_page_num):
    """Draw bounding boxes, extract images, save boxed PNG."""
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    images_dir = os.path.join(page_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    img_idx = 0
    for ref in refs:
        try:
            result = extract_coords(ref)
            if not result:
                continue
            label_type, points_list = result
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)

            for points in points_list:
                x1 = int(points[0] / 999 * img_w)
                y1 = int(points[1] / 999 * img_h)
                x2 = int(points[2] / 999 * img_w)
                y2 = int(points[3] / 999 * img_h)

                if label_type == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(os.path.join(images_dir, f"page{real_page_num}_{img_idx}.jpg"))
                        img_idx += 1
                    except Exception:
                        pass

                width = 4 if label_type == "title" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                tx, ty = x1, max(0, y1 - 18)
                bbox = draw.textbbox((0, 0), label_type, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.rectangle([tx, ty, tx + tw, ty + th], fill=(255, 255, 255, 220))
                draw.text((tx, ty), label_type, font=font, fill=color)
        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    boxed_path = os.path.join(page_dir, f"page_{real_page_num}_boxed.png")
    img_draw.save(boxed_path)
    return boxed_path


def clean_output(content, matches_img, matches_other):
    """Remove ref/det tokens, replace image refs with markdown."""
    for idx, m in enumerate(matches_img):
        content = content.replace(m, f"![](images/page{pagenum}_{idx}.jpg)\n")
    for m in matches_other:
        content = content.replace(m, "")
    content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    content = re.sub(r"\n\n\n\n", "\n\n", content)
    content = re.sub(r"\n\n\n", "\n\n", content)
    content = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", content)
    return content.strip()


pagenum = 0


def main():
    global pagenum

    # --- Extract pages from PDF ---
    print("[1/3] Extracting pages from PDF...")
    print(f"  PDF: {PDF_PATH}")
    images_map = {}
    for pi in PAGES_TO_PROCESS:
        img = pdf_page_to_image(PDF_PATH, pi, DPI)
        images_map[pi] = img
        print(f"  Page {pi + 1}: {img.size[0]}x{img.size[1]} px")
    print()

    # --- Load model ---
    print("[2/3] Loading DeepSeek OCR model (4-bit quantized)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_safetensors=True,
        device_map="auto", quantization_config=quantization_config,
        attn_implementation="eager",
    )
    allocated = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    print(f"  Model loaded. GPU memory: {allocated:.2f} GB")
    print()

    # --- Process each page ---
    print("[3/3] Running OCR on pages 21 and 23...")
    for pi in PAGES_TO_PROCESS:
        real_page_num = pi + 1
        pagenum = real_page_num
        page_dir = os.path.join(OUTPUT_BASE, f"page_{real_page_num}")
        os.makedirs(page_dir, exist_ok=True)

        # Save page image temporarily
        temp_img_path = os.path.join(page_dir, f"_temp_page_{real_page_num}.png")
        images_map[pi].save(temp_img_path)

        print(f"  Processing page {real_page_num}...")
        output_text = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=temp_img_path,
            output_path=page_dir,
            base_size=1024, image_size=640, crop_mode=True,
            eval_mode=True, test_compress=False, save_results=False,
        )

        if output_text is None:
            print(f"    WARNING: No output for page {real_page_num}")
            continue

        # Parse bounding boxes
        matches_ref, matches_img, matches_other = re_match(output_text)

        # Draw boxes + extract images
        boxed_path = draw_boxes_and_extract(
            images_map[pi].copy(), matches_ref, page_dir, real_page_num
        )

        # Clean markdown (use custom clean that references correct pagenum)
        content = output_text
        for idx, m in enumerate(matches_img):
            content = content.replace(m, f"![](images/page{real_page_num}_{idx}.jpg)\n")
        for m in matches_other:
            content = content.replace(m, "")
        content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
        content = re.sub(r"\n\n\n\n", "\n\n", content)
        content = re.sub(r"\n\n\n", "\n\n", content)
        content = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", content)
        md_content = content.strip()

        # Save markdown
        md_path = os.path.join(page_dir, f"page_{real_page_num}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Save HTML (basic conversion)
        import html as html_mod
        paragraphs = md_content.split("\n\n")
        parts = [
            "<!DOCTYPE html>", '<html lang="tr">', "<head>", '<meta charset="UTF-8">',
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
                    parts.append("<tr>" + "".join(f"<td>{c.strip()}</td>" for c in cells) + "</tr>")
                continue
            else:
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
                content_html = html_mod.escape(para)
                content_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content_html)
                content_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content_html)
                content_html = re.sub(r"!\[(.*?)\]\((.+?)\)", r'<img src="\2" alt="\1">', content_html)
                content_html = re.sub(r"\[(.*?)\]\((.+?)\)", r'<a href="\2">\1</a>', content_html)
                parts.append(f"<p>{content_html}</p>")
        if in_table:
            parts.append("</table>")
        parts.extend(["</body>", "</html>"])
        html_path = os.path.join(page_dir, f"page_{real_page_num}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        # Save original
        orig_path = os.path.join(page_dir, f"page_{real_page_num}_original.png")
        images_map[pi].save(orig_path)

        # Cleanup temp
        try:
            os.remove(temp_img_path)
        except Exception:
            pass

        n_boxes = len(matches_ref)
        n_imgs = len(matches_img)
        print(f"    boxes: {n_boxes} | extracted images: {n_imgs}")
        print(f"    -> {md_path}")
        print(f"    -> {html_path}")
        print(f"    -> {boxed_path}")
        print(f"    -> {orig_path}")
        print()

    # --- Summary ---
    print("=" * 60)
    print("Output structure:")
    for pi in PAGES_TO_PROCESS:
        rpn = pi + 1
        d = os.path.join(OUTPUT_BASE, f"page_{rpn}")
        print(f"  {d}/")
        print(f"    +-- page_{rpn}.md")
        print(f"    +-- page_{rpn}.html")
        print(f"    +-- page_{rpn}_boxed.png")
        print(f"    +-- page_{rpn}_original.png")
        print(f"    +-- images/")
        img_dir = os.path.join(d, "images")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                print(f"         +-- {f}")
    print("Done!")


if __name__ == "__main__":
    main()
