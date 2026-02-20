from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

char_list: str = (
    "人一日大年出本中子見国言上分生手自行者二間事思時気会十家女三前的方入小地合後目長場代私下立部学物月田"
    "何来彼話体動社知理山内同心発高実作当新世今書度明五戦力名金性対意用男主通関文屋感郎業定政持道外取所現"
)


def sort_img_paths_by_char(img_paths: Sequence[Path], char_list: str) -> List[Path]:
    char_to_path: Dict[str, Path] = {}
    for p in img_paths:
        # 例: "gen_人.png" -> "人"
        parts = p.stem.split("_")
        if len(parts) >= 2:
            char = parts[1]
            char_to_path[char] = p

    return [char_to_path[c] if c in char_to_path else Path() for c in char_list]


def save_ref_grid_pdf_fixed(
    refs: Sequence[Image.Image],
    imgs: Sequence[Image.Image],
    output_path: Path,
    page_size: Tuple[int, int],
    margin: int,
    # ---- text ----
    text_size: int,
    ref_text: str,
    ref_text_pos: Tuple[int, int],
    gen_text: str,
    gen_text_pos: Tuple[int, int],
    # ---- ref image ----
    ref_pos: Tuple[int, int],
    ref_size: Tuple[int, int],
    ref_border: int,
    ref_color: Tuple[int, int, int],
    # ---- grid ----
    grid_pos: Tuple[int, int],
    cell: Tuple[int, int],
    cols: int,
    rows: int,
    grid_border: int,
    grid_color: Tuple[int, int, int],
) -> None:
    """
    Layout (fixed, no auto-calc):
        ref text
        ref images (horizontally)
        gen text
        generated image grid
    """
    canvas: Image.Image = Image.new("RGB", page_size, "white")
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)

    # Font (best-effort)
    try:
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype(
            "DejaVuSans.ttf", text_size
        )
    except OSError:
        font = ImageFont.load_default()

    # ======================
    # 1) REF TEXT
    # ======================
    draw.text(ref_text_pos, ref_text, fill="black", font=font)

    # ======================
    # 2) REF IMAGES
    # ======================
    for i, r in enumerate(refs):
        r_resized = r.convert("RGB").resize(ref_size, Image.BILINEAR)
        rx = ref_pos[0] + i * (ref_size[0] + margin)
        ry = ref_pos[1]

        canvas.paste(r_resized, (rx, ry))
        draw.rectangle(
            (
                rx - ref_border // 2,
                ry - ref_border // 2,
                rx + ref_size[0] + ref_border // 2,
                ry + ref_size[1] + ref_border // 2,
            ),
            outline=ref_color,
            width=ref_border,
        )

    # ======================
    # 3) GEN TEXT
    # ======================
    draw.text(gen_text_pos, gen_text, fill="black", font=font)

    # ======================
    # 4) GEN GRID
    # ======================
    gx, gy = grid_pos
    cw, ch = cell
    max_n = cols * rows

    for i, im in enumerate(imgs[:max_n]):
        rr, cc = divmod(i, cols)
        x = gx + cc * cw
        y = gy + rr * ch

        im_resized = im.convert("RGB").resize((cw, ch), Image.BILINEAR)
        canvas.paste(im_resized, (x, y))

        draw.rectangle(
            (
                x - grid_border // 2,
                y - grid_border // 2,
                x + cw + grid_border // 2,
                y + ch + grid_border // 2,
            ),
            outline=grid_color,
            width=grid_border,
        )

    canvas.save(output_path, "PDF")
    print(f"Saved PDF: {output_path}")


if __name__ == "__main__":
    output_dir = Path("/home/yamamoto/workspace/FontDiffuser/outputs")
    img_dir = output_dir / "sample_content_image_100_ref_Noto_Serif_TC_Bold"
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # refs
    ref_image_paths: List[Path] = sorted(img_dir.glob("ref_*.png"))
    print(f"Found {len(ref_image_paths)} ref images.")
    if len(ref_image_paths) <= 5:
        ref_size = 400
    elif len(ref_image_paths) == 8:
        ref_image_paths = ref_image_paths[:8]
        ref_size = 275
    else: 
        ref_size = 350
        
    ref_images: List[Image.Image] = [
        Image.open(p).convert("RGB") for p in ref_image_paths
    ]

    # gens (sorted by char_list)
    gen_image_paths: List[Path] = sorted(img_dir.glob("gen_*.png"))
    gen_image_paths = sort_img_paths_by_char(gen_image_paths, char_list)
    gen_images: List[Image.Image] = [
        Image.open(p).convert("RGB")
        if p
        else Image.new("RGB", (ref_images[0].height, ref_images[0].height), "white")
        for p in gen_image_paths
        for p in gen_image_paths
    ]

    output_path = pdf_dir / f"{img_dir.name}.pdf"

    save_ref_grid_pdf_fixed(
        refs=ref_images,
        imgs=gen_images,
        output_path=output_path,
        page_size=(2480, 3508),  # A4 300dpi
        margin=20,
        # ---- text ----
        text_size=100,
        ref_text="Reference",
        ref_text_pos=(40, 0),
        gen_text="Generated",
        gen_text_pos=(40, 650),
        # ---- ref image ----
        ref_pos=(40, 150),
        ref_size=(ref_size, ref_size),
        ref_border=12,
        ref_color=(255, 0, 0),
        # ---- grid ----
        grid_pos=(40, 800),
        cell=(225, 225),
        cols=10,
        rows=10,
        grid_border=6,
        grid_color=(0, 0, 0),
    )
