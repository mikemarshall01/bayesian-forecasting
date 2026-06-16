#!/usr/bin/env python3
"""Watermark chart PNGs with the GitHub mark + a handle, semi-transparent, lower-right.

Self-contained + idempotent. The mark (gh_181717.png) must sit next to this script.
Writes a PNG text marker so re-runs (and CI) never double-stamp.

  python watermark.py "assets/*.png" --inplace            # apply, skip already-marked
  python watermark.py "assets/*.png"                       # write <name>_WM.png copies
Options: --handle (mikemarshall01) --alpha (0.45) --pad-bottom (58) --mark-h (34) --font (26)
"""
import sys, glob, argparse, os, json, base64, io
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = os.path.join(HERE, "gh_181717.png")          # GitHub mark, brand colour #181717
GH = (24, 23, 23)                                   # GitHub brand black #181717
MARKER_KEY, MARKER_VAL = "watermark", "mikemarshall01"
FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
try:                                                # prefer matplotlib's bundled DejaVuSans, found at runtime
    import matplotlib
    FONT_CANDIDATES.insert(0, os.path.join(os.path.dirname(matplotlib.__file__),
                                           "mpl-data/fonts/ttf/DejaVuSans.ttf"))
except Exception:
    pass

def load_font(px):
    for p in FONT_CANDIDATES:
        try: return ImageFont.truetype(p, px)
        except Exception: pass
    return ImageFont.load_default()

def already_marked(path):
    try:
        return Image.open(path).info.get(MARKER_KEY) == MARKER_VAL
    except Exception:
        return False

def gh_mark(height):
    m = Image.open(MARK).convert("RGBA")
    w = int(m.size[0] * height / m.size[1])
    return m.resize((w, height), Image.LANCZOS)

def watermark_pil(base, handle="mikemarshall01", alpha=0.45, pad=18, pad_bottom=58, mark_h=34, font_px=26):
    """Stamp a PIL image (RGBA in) and return an RGBA image. Core placement logic."""
    base = base.convert("RGBA")
    W, H = base.size
    mark = gh_mark(mark_h)
    font = load_font(font_px)
    text = " " + handle
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tb = tmp.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ov_w = mark.size[0] + tw + 6
    ov_h = max(mark_h, th) + 4
    ov = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    ov.alpha_composite(mark, (0, (ov_h - mark_h) // 2))
    ImageDraw.Draw(ov).text((mark.size[0] + 6, (ov_h - th) // 2 - tb[1]), text, font=font, fill=GH + (255,))
    ov.putalpha(ov.split()[3].point(lambda v: int(v * alpha)))
    base.alpha_composite(ov, (W - ov_w - pad, H - ov_h - pad_bottom))
    return base

def watermark_notebook(path, handle="mikemarshall01", alpha=0.45, pad_bottom=58, mark_h=34, font_px=26, skip_marked=True):
    """Stamp the embedded image/png outputs inside a .ipynb (what shows when you open it). Idempotent."""
    j = json.load(open(path))
    n = 0
    for c in j.get("cells", []):
        for o in c.get("outputs", []):
            data = o.get("data", {})
            b = data.get("image/png")
            if not b: continue
            raw = base64.b64decode(b if isinstance(b, str) else "".join(b))
            im = Image.open(io.BytesIO(raw))
            if skip_marked and im.info.get(MARKER_KEY) == MARKER_VAL: continue
            stamped = watermark_pil(im, handle, alpha, pad_bottom=pad_bottom, mark_h=mark_h, font_px=font_px).convert("RGB")
            buf = io.BytesIO(); meta = PngInfo(); meta.add_text(MARKER_KEY, MARKER_VAL)
            stamped.save(buf, format="PNG", pnginfo=meta)
            data["image/png"] = base64.b64encode(buf.getvalue()).decode("ascii")
            n += 1
    if n: json.dump(j, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return n

def watermark(path, handle="mikemarshall01", alpha=0.45, pad=18, pad_bottom=58,
              mark_h=34, font_px=26, inplace=False):
    base = Image.open(path).convert("RGBA")
    W, H = base.size
    mark = gh_mark(mark_h)
    font = load_font(font_px)
    text = " " + handle
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    tb = tmp.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ov_w = mark.size[0] + tw + 6
    ov_h = max(mark_h, th) + 4
    ov = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    ov.alpha_composite(mark, (0, (ov_h - mark_h) // 2))
    d = ImageDraw.Draw(ov)
    d.text((mark.size[0] + 6, (ov_h - th) // 2 - tb[1]), text, font=font, fill=GH + (255,))
    ov.putalpha(ov.split()[3].point(lambda v: int(v * alpha)))
    base.alpha_composite(ov, (W - ov_w - pad, H - ov_h - pad_bottom))
    meta = PngInfo(); meta.add_text(MARKER_KEY, MARKER_VAL)
    out = path if inplace else path.rsplit(".", 1)[0] + "_WM.png"
    base.convert("RGB").save(out, pnginfo=meta)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--handle", default="mikemarshall01")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--pad-bottom", type=int, default=58)
    ap.add_argument("--mark-h", type=int, default=34)
    ap.add_argument("--font", type=int, default=26)
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--skip-marked", action="store_true", help="skip images already watermarked")
    ap.add_argument("--notebooks", action="store_true", help="target is .ipynb files; stamp embedded image/png outputs")
    a = ap.parse_args()
    if a.notebooks:
        tot = 0
        for f in sorted(glob.glob(a.target)):
            if ".ipynb_checkpoints" in f: continue
            k = watermark_notebook(f, a.handle, a.alpha, a.pad_bottom, a.mark_h, a.font, a.skip_marked)
            if k: print(f"  {f}: stamped {k} embedded images")
            tot += k
        print(f"notebook outputs watermarked: {tot}")
    else:
        done = skipped = 0
        for f in sorted(glob.glob(a.target)):
            if f.endswith("_WM.png"): continue
            if a.skip_marked and already_marked(f): skipped += 1; continue
            watermark(f, a.handle, a.alpha, pad_bottom=a.pad_bottom, mark_h=a.mark_h, font_px=a.font, inplace=a.inplace)
            done += 1
        print(f"watermarked {done}, skipped {skipped} (already marked)")
