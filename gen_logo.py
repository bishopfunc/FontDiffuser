from pathlib import Path

from PIL import Image

FONT_DIR = Path(
    "/home/yamamoto/workspace/FontDiffuser/outputs/sample_content_image_100_ref_Noto_Serif_TC_Bold_few_shot_logo"
)
OUT_PATH = Path("deep_font_logo.png")
LOGO_TEXT = "深度字体"

cell_size = 256
coordinates = []
for i in range(2):
    for j in range(2):
        x = j * cell_size + cell_size // 2
        y = i * cell_size + cell_size // 2
        coordinates.append((x, y))

# これを4個並べる
base_image = Image.new("RGB", (cell_size * 2, cell_size * 2), (255, 255, 255))
for idx, (x, y) in enumerate(coordinates):
    char = LOGO_TEXT[idx]
    image_path = FONT_DIR / f"gen_{char}.png"
    if image_path.exists():
        char_image = Image.open(image_path).convert("RGBA")
        char_image = char_image.resize((cell_size, cell_size), resample=Image.BICUBIC)
        base_image.paste(
            char_image, (x - cell_size // 2, y - cell_size // 2), char_image
        )
    else:
        print(f"Warning: Image not found for character '{char}': {image_path}")

base_image.save(OUT_PATH)
print(f"Logo saved to {OUT_PATH}")
