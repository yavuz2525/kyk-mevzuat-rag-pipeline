import os, re, io, numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(BASE_PATH, "output")
TEMP_PAGES_DIR = os.path.join(OUTPUT_BASE, "temp_pages")
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

os.makedirs(TEMP_PAGES_DIR, exist_ok=True)

print("[1/3] Extracting pages 7-14...")
doc = fitz.open(os.path.join(BASE_PATH, "7.5.38510.pdf"))
total = doc.page_count
images = []
for i in range(6, total):
    page = doc[i]
    zoom = 144 / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pixmap.tobytes("png")))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    images.append(img)
    img.save(os.path.join(TEMP_PAGES_DIR, f"page_{i+1}.png"))
    print(f"  Page {i+1}: {img.size[0]}x{img.size[1]}")
doc.close()
print()

print("[2/3] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
qconfig = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
model = AutoModel.from_pretrained(
    MODEL_NAME, trust_remote_code=True, use_safetensors=True,
    device_map="auto", quantization_config=qconfig, attn_implementation="eager",
)
print(f"  GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB\n")


def re_match(text):
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    m_img, m_other = [], []
    for m in matches:
        if "<|ref|>image<|/ref|>" in m[0]:
            m_img.append(m[0])
        else:
            m_other.append(m[0])
    return matches, m_img, m_other


def extract_coords(ref_text):
    try:
        return (ref_text[1], eval(ref_text[2]))
    except Exception:
        return None


def draw_boxes(image, refs, page_dir, page_num):
    iw, ih = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    img_dir = os.path.join(page_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    img_idx = 0
    for ref in refs:
        try:
            r = extract_coords(ref)
            if not r:
                continue
            lt, pts = r
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            ca = color + (25,)
            for p in pts:
                x1 = int(p[0] / 999 * iw)
                y1 = int(p[1] / 999 * ih)
                x2 = int(p[2] / 999 * iw)
                y2 = int(p[3] / 999 * ih)
                if lt == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(os.path.join(img_dir, f"{page_num}_{img_idx}.jpg"))
                        img_idx += 1
                    except Exception:
                        pass
                w = 4 if lt == "title" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=w)
                draw2.rectangle([x1, y1, x2, y2], fill=ca, outline=(0, 0, 0, 0), width=1)
                tx, ty = x1, max(0, y1 - 16)
                bb = draw.textbbox((0, 0), lt, font=font)
                draw.rectangle(
                    [tx, ty, tx + bb[2] - bb[0], ty + bb[3] - bb[1]],
                    fill=(255, 255, 255, 220),
                )
                draw.text((tx, ty), lt, font=font, fill=color)
        except Exception:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    bp = os.path.join(page_dir, f"page_{page_num}_boxed.png")
    img_draw.save(bp)


def md_to_html(md):
    import html as hm
    parts = [
        "<!DOCTYPE html>",
        '<html lang="tr">',
        "<head>",
        '<meta charset="UTF-8">',
        "<style>",
        "body{font-family:system-ui;max-width:800px;margin:40px auto;padding:20px;line-height:1.6;color:#333}",
        "h1{border-bottom:2px solid #eee;padding-bottom:8px}",
        "h2{border-bottom:1px solid #eee;padding-bottom:6px}",
        "table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ddd;padding:8px}th{background:#f5f5f5}",
        "img{max-width:100%}",
        "</style></head><body>",
    ]
    for para in md.split("\n\n"):
        if not para.strip():
            continue
        if para.startswith("### "):
            parts.append(f"<h3>{hm.escape(para[4:])}</h3>")
        elif para.startswith("## "):
            parts.append(f"<h2>{hm.escape(para[3:])}</h2>")
        elif para.startswith("# "):
            parts.append(f"<h1>{hm.escape(para[2:])}</h1>")
        else:
            c = hm.escape(para)
            c = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", c)
            c = re.sub(r"!\[(.*?)\]\((.+?)\)", r'<img src="\2" alt="\1">', c)
            parts.append(f"<p>{c}</p>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)


def clean_output(content, m_img, m_other):
    for idx, m in enumerate(m_img):
        content = content.replace(m, f"![]({idx}.jpg)\n")
    for m in m_other:
        content = content.replace(m, "")
    content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    content = re.sub(r"\n\n\n+", "\n\n", content)
    content = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", content)
    return content.strip()


print("[3/3] Processing pages 7-14...")
for idx, img in enumerate(images):
    page_num = idx + 7
    page_dir = os.path.join(OUTPUT_BASE, f"page_{page_num}")
    os.makedirs(page_dir, exist_ok=True)
    img_path = os.path.join(TEMP_PAGES_DIR, f"page_{page_num}.png")

    print(f"  Page {page_num}...", end=" ", flush=True)
    output_text = model.infer(
        tokenizer, prompt=PROMPT, image_file=img_path, output_path=page_dir,
        base_size=1024, image_size=640, crop_mode=True,
        eval_mode=True, test_compress=False, save_results=False,
    )
    if not output_text:
        print("no output")
        continue

    matches_ref, matches_img, matches_other = re_match(output_text)
    draw_boxes(img.copy(), matches_ref, page_dir, page_num)

    md_content = clean_output(output_text, matches_img, matches_other)
    with open(os.path.join(page_dir, f"page_{page_num}.md"), "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(os.path.join(page_dir, f"page_{page_num}.html"), "w", encoding="utf-8") as f:
        f.write(md_to_html(md_content))
    img.save(os.path.join(page_dir, f"page_{page_num}_original.png"))

    print(f"boxes:{len(matches_ref)} imgs:{len(matches_img)}")

print("\nDone! All 14 pages processed.")
