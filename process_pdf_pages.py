"""
DeepSeek OCR PDF Page Processor
Processes the first 3 pages of a PDF with DeepSeek OCR,
creating per-page directories with markdown, HTML, boxed images, and extracted figures.

Usage:
  python process_pdf_pages.py

Requirements:
  - DeepSeek OCR model downloaded (deepseek-ai/DeepSeek-OCR)
  - CUDA GPU with sufficient VRAM (16GB+)
  - vllm, transformers, flash-attn installed
  - PyMuPDF, Pillow, markdown installed
"""

import os
import sys
import re
import fitz
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- Configuration ---
PDF_PATH = "7.5.38510.pdf"
OUTPUT_BASE = "output"
NUM_PAGES = 3
DPI = 144

# DeepSeek OCR settings (matching config.py defaults)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

os.makedirs(OUTPUT_BASE, exist_ok=True)


def pdf_page_to_image(pdf_path, page_num, dpi=144):
    """Convert a single PDF page to a PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    doc.close()
    return img


def re_match(text):
    """Parse OCR output for bounding box references."""
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


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """Extract label type and coordinate list from a ref match."""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
        return (label_type, cor_list)
    except Exception:
        return None


def draw_bounding_boxes(image, refs, output_dir, page_idx):
    """Draw bounding boxes on image, extract embedded figures, save result."""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    img_idx = 0

    for ref in refs:
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if not result:
                continue
            label_type, points_list = result
            color = (
                np.random.randint(0, 200),
                np.random.randint(0, 200),
                np.random.randint(0, 255),
            )
            color_a = color + (20,)

            for points in points_list:
                x1 = int(points[0] / 999 * image_width)
                y1 = int(points[1] / 999 * image_height)
                x2 = int(points[2] / 999 * image_width)
                y2 = int(points[3] / 999 * image_height)

                # Extract embedded images/figures
                if label_type == "image":
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        cropped.save(os.path.join(images_dir, f"{page_idx}_{img_idx}.jpg"))
                        img_idx += 1
                    except Exception:
                        pass

                # Draw bounding box
                try:
                    width = 4 if label_type == "title" else 2
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a,
                                    outline=(0, 0, 0, 0), width=1)

                    text_x, text_y = x1, max(0, y1 - 18)
                    bbox = draw.textbbox((0, 0), label_type, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.rectangle(
                        [text_x, text_y, text_x + tw, text_y + th],
                        fill=(255, 255, 255, 220),
                    )
                    draw.text((text_x, text_y), label_type, font=font, fill=color)
                except Exception:
                    pass
        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    boxed_path = os.path.join(output_dir, f"page_{page_idx + 1}_boxed.png")
    img_draw.save(boxed_path)
    return boxed_path


def convert_md_to_html(md_text):
    """Convert markdown text to basic HTML."""
    import html as html_mod

    paragraphs = md_text.split("\n\n")
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="UTF-8">',
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; "
        "color: #333; }",
        "h1 { font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 8px; }",
        "h2 { font-size: 1.4em; border-bottom: 1px solid #eee; padding-bottom: 6px; }",
        "h3 { font-size: 1.2em; }",
        "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
        "td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f5f5f5; }",
        "img { max-width: 100%; height: auto; }",
        "code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
        "pre { background: #f4f4f4; padding: 16px; border-radius: 6px; overflow-x: auto; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    in_table = False
    for para in paragraphs:
        if not para.strip():
            continue
        lines = para.split("\n")

        # Table detection
        if lines[0].startswith("|") and lines[0].endswith("|"):
            if not in_table:
                html_parts.append("<table>")
                in_table = True
            for line in lines:
                if line.startswith("|-") or line.startswith("| --"):
                    continue
                cells = line.split("|")[1:-1]
                tag = "th" if in_table and len(html_parts) > 0 else "td"
                html_parts.append(
                    "<tr>"
                    + "".join(f"<{tag}>{c.strip()}</{tag}>" for c in cells)
                    + "</tr>"
                )
            continue
        else:
            if in_table:
                html_parts.append("</table>")
                in_table = False

        # Headers
        if para.startswith("### "):
            html_parts.append(f"<h3>{html_mod.escape(para[4:])}</h3>")
        elif para.startswith("## "):
            html_parts.append(f"<h2>{html_mod.escape(para[3:])}</h2>")
        elif para.startswith("# "):
            html_parts.append(f"<h1>{html_mod.escape(para[2:])}</h1>")
        else:
            # Process inline formatting
            content = html_mod.escape(para)
            # Bold
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            # Italic
            content = re.sub(r"\*(.+?)\*", r"<em>\1</em>", content)
            # Images
            content = re.sub(
                r"!\[(.*?)\]\((.+?)\)",
                r'<img src="\2" alt="\1">',
                content,
            )
            # Links
            content = re.sub(
                r"\[(.*?)\]\((.+?)\)",
                r'<a href="\2">\1</a>',
                content,
            )
            html_parts.append(f"<p>{content}</p>")

    if in_table:
        html_parts.append("</table>")
    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


def clean_markdown(content):
    """Replace ref tokens with proper markdown image references, clean up artifacts."""
    # Replace image refs with markdown syntax
    for match_text in matches_images_global:
        content = content.replace(match_text, "")

    content = content.replace("\\coloneqq", ":=")
    content = content.replace("\\eqqcolon", "=:")
    content = re.sub(r"\n\n\n\n", "\n\n", content)
    content = re.sub(r"\n\n\n", "\n\n", content)
    # Remove any remaining ref/det tokens
    content = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", content, flags=re.DOTALL)
    return content.strip()


matches_images_global = []


def process_with_deepseek_ocr(images):
    """
    Run DeepSeek OCR on the given images using vLLM.
    Returns list of (raw_output_text, image_obj) tuples.
    """
    import torch

    if torch.version.cuda == "11.8":
        os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from deepseek_ocr import DeepseekOCRForCausalLM
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm import LLM, SamplingParams
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
    from config import CROP_MODE as CROP

    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=min(8, len(images)),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )

    logits_processor = NoRepeatNGramLogitsProcessor(
        ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822}
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=[logits_processor],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    processor = DeepseekOCRProcessor()
    batch_inputs = []
    for img in images:
        cache_item = {
            "prompt": PROMPT,
            "multi_modal_data": {
                "image": processor.tokenize_with_images(
                    images=[img], bos=True, eos=True, cropping=CROP
                )
            },
        }
        batch_inputs.append(cache_item)

    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)
    results = []
    for output, img in zip(outputs_list, images):
        text = output.outputs[0].text
        if "<｜end▁of▁sentence｜>" in text:
            text = text.replace("<｜end▁of▁sentence｜>", "")
        results.append((text, img))

    return results


def process_with_huggingface_ocr(images):
    """
    Run DeepSeek OCR on the given images using HuggingFace Transformers.
    Falls back from vLLM if not available.
    """
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    results = []
    for img in images:
        temp_output = os.path.join(OUTPUT_BASE, "_temp")
        os.makedirs(temp_output, exist_ok=True)

        # Save image temporarily
        temp_img_path = os.path.join(temp_output, "_temp_page.png")
        img.save(temp_img_path)

        res = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=temp_img_path,
            output_path=temp_output,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=True,
            test_compress=False,
        )

        # Read the result file
        result_mmd = os.path.join(temp_output, "result.mmd")
        if os.path.exists(result_mmd):
            with open(result_mmd, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = str(res) if res else ""

        results.append((text, img))

    return results


def main():
    print(f"PDF: {PDF_PATH}")
    print(f"Output: {OUTPUT_BASE}/")
    print(f"Processing first {NUM_PAGES} pages...")
    print()

    # Step 1: Extract page images
    print("[1/4] Extracting pages from PDF...")
    images = []
    for i in range(NUM_PAGES):
        img = pdf_page_to_image(PDF_PATH, i, dpi=DPI)
        images.append(img)
        print(f"  Page {i + 1}: {img.size[0]}x{img.size[1]} px")
    print()

    # Step 2: Run OCR
    print("[2/4] Running DeepSeek OCR...")
    try:
        # Try vLLM first (faster)
        results = process_with_deepseek_ocr(images)
        print("  Using vLLM inference")
    except Exception as e:
        print(f"  vLLM not available ({e}), trying HuggingFace...")
        try:
            results = process_with_huggingface_ocr(images)
            print("  Using HuggingFace inference")
        except Exception as e2:
            print(f"  ERROR: Could not load DeepSeek OCR model: {e2}")
            print("  Please ensure:")
            print("    1. deepseek-ai/DeepSeek-OCR model is downloaded")
            print("    2. vllm/transformers + flash-attn are installed")
            print("    3. CUDA GPU is available")
            print()
            print("  Extracted page images are available in the output directory.")
            sys.exit(1)
    print()

    # Step 3: Process per-page outputs
    print("[3/4] Generating per-page outputs...")
    global matches_images_global

    for page_idx, (content, img) in enumerate(results):
        page_dir = os.path.join(OUTPUT_BASE, f"page_{page_idx + 1}")
        os.makedirs(page_dir, exist_ok=True)

        # Parse OCR output
        matches_ref, matches_img, matches_other = re_match(content)

        # Draw bounding boxes + extract images
        boxed_path = draw_bounding_boxes(img.copy(), matches_ref, page_dir, page_idx)

        # Build clean markdown with image references
        md_content = content
        for idx, match_img in enumerate(matches_img):
            md_content = md_content.replace(
                match_img, f"![](images/{page_idx}_{idx}.jpg)\n"
            )
        for match_other in matches_other:
            md_content = md_content.replace(match_other, "")

        md_content = clean_markdown(md_content)

        # Remove page split marker if present
        md_content = md_content.replace("<--- Page Split --->", "").strip()

        # Remove only the model's internal ref/det tokens, keep markdown images
        md_content = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", md_content)

        # Save markdown
        md_path = os.path.join(page_dir, f"page_{page_idx + 1}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Save HTML
        html_content = convert_md_to_html(md_content)
        html_path = os.path.join(page_dir, f"page_{page_idx + 1}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save original page image
        orig_path = os.path.join(page_dir, f"page_{page_idx + 1}_original.png")
        img.save(orig_path)

        # Summary
        n_images = len(matches_img)
        n_boxes = len(matches_ref)
        print(f"  page_{page_idx + 1}/")
        print(f"    boxes: {n_boxes} | extracted images: {n_images}")
        print(f"    -> {md_path}")
        print(f"    -> {html_path}")
        print(f"    -> {boxed_path}")
        print(f"    -> {orig_path}")

    # Step 4: Cleanup
    print()
    print("[4/4] Done. Output structure:")
    for i in range(NUM_PAGES):
        page_dir = os.path.join(OUTPUT_BASE, f"page_{i + 1}")
        print(f"  {page_dir}/")
        print(f"    ├── page_{i + 1}.md")
        print(f"    ├── page_{i + 1}.html")
        print(f"    ├── page_{i + 1}_boxed.png")
        print(f"    ├── page_{i + 1}_original.png")
        print(f"    └── images/")
        img_dir = os.path.join(page_dir, "images")
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                print(f"         └── {fname}")


if __name__ == "__main__":
    main()
