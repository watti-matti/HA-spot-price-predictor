"""Generate transparent, trimmed WM brand icons for Home Assistant.

Shared Watti-matti brand mark (the gold "WM" hexagon badge) — the same
icon used by the HA Energy Needs Planner integration, so the two products
read as one family in Home Assistant.

Home Assistant's integration-card icon comes from the central
``home-assistant/brands`` repo, which requires square PNGs with a
**transparent** background, trimmed to the content. The source WM badge
is a gold hexagon on an opaque black field, so this script:

1. Flood-fills the *exterior* near-black background to alpha 0, starting
   from the four corners. The hexagon's interior black is enclosed by the
   gold border, so the flood never reaches it — the badge body is kept.
2. Trims to the badge bounding box and pads to a tight square.
3. Exports ``icon.png`` (256×256) and ``icon@2x.png`` (512×512) into the
   integration folder, its ``brand/`` folder, and the brands submission
   bundle under ``brands/custom_integrations/spot_price_predictor/``.

Usage::

    python scripts/make_brand_icons.py [SOURCE_PNG]

SOURCE_PNG defaults to
``custom_components/spot_price_predictor/brand/watti-matti_logo.png``.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "spot_price_predictor"
DEFAULT_SRC = ROOT / f"custom_components/{DOMAIN}/brand/watti-matti_logo.png"
_BG_MAX = 48  # a pixel is "background" if every RGB channel is below this


def _is_bg(p: tuple[int, int, int, int]) -> bool:
    r, g, b, _ = p
    return r < _BG_MAX and g < _BG_MAX and b < _BG_MAX


def make_transparent(im: Image.Image) -> Image.Image:
    """Flood-fill the exterior near-black background to transparent."""
    im = im.convert("RGBA")
    w, h = im.size
    px = im.load()
    seen = [[False] * w for _ in range(h)]
    dq: deque[tuple[int, int]] = deque()
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if _is_bg(px[sx, sy]):
            dq.append((sx, sy))
            seen[sy][sx] = True
    while dq:
        x, y = dq.popleft()
        r, g, b, _ = px[x, y]
        px[x, y] = (r, g, b, 0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and _is_bg(
                px[nx, ny]
            ):
                seen[ny][nx] = True
                dq.append((nx, ny))
    return im


def square_trim(im: Image.Image) -> Image.Image:
    """Trim to content and pad to a tight transparent square."""
    bbox = im.getbbox()
    trimmed = im.crop(bbox)
    tw, th = trimmed.size
    side = max(tw, th)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(trimmed, ((side - tw) // 2, (side - th) // 2), trimmed)
    return canvas


def main(src: Path) -> None:
    base = square_trim(make_transparent(Image.open(src)))
    icon1x = base.resize((256, 256), Image.LANCZOS)
    icon2x = base.resize((512, 512), Image.LANCZOS)

    cc = ROOT / f"custom_components/{DOMAIN}"
    brands = ROOT / f"brands/custom_integrations/{DOMAIN}"
    targets_1x = [
        cc / "icon.png",
        cc / "brand/icon.png",
        cc / "brand/logo.png",
        brands / "icon.png",
    ]
    targets_2x = [
        cc / "icon@2x.png",
        cc / "brand/icon@2x.png",
        brands / "icon@2x.png",
    ]
    for p in targets_1x:
        p.parent.mkdir(parents=True, exist_ok=True)
        icon1x.save(p)
    for p in targets_2x:
        p.parent.mkdir(parents=True, exist_ok=True)
        icon2x.save(p)
    print(f"wrote {len(targets_1x)} × 256px and {len(targets_2x)} × 512px icons")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC)
