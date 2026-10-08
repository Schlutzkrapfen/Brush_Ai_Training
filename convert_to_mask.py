"""
Step 1: Label Studio export.json  ->  multi-class mask PNGs + classes.json

Export from Label Studio as plain "JSON" (not JSON-MIN).
Put the images in data/images/. Run:  python convert_to_mask.py
"""
import json, os
import cv2, numpy as np
from label_studio_converter.brush import decode_rle

EXPORT_FILE = "export.json"
IMG_DIR = "data/images"
OUT_DIR = "data/masks"

os.makedirs(OUT_DIR, exist_ok=True)

with open(EXPORT_FILE) as f:
    tasks = json.load(f)


def find_image(path):
    """Match the Label Studio image path to a file in IMG_DIR (with/without upload prefix)."""
    base = os.path.basename(path)
    for cand in (base, base.split("-", 1)[-1]):
        if os.path.exists(os.path.join(IMG_DIR, cand)):
            return cand
    return None


def brush_results(task):
    for ann in task.get("annotations", []):
        if ann.get("was_cancelled"):
            continue
        for r in ann.get("result", []):
            if r.get("type") == "brushlabels" and r.get("value", {}).get("brushlabels"):
                yield r


# 1. class names come from the data itself (no typing, no typos)
found = sorted({r["value"]["brushlabels"][0] for t in tasks for r in brush_results(t)})
class_id = {n: i + 1 for i, n in enumerate(found)}   # 0 = background
with open("classes.json", "w") as f:
    json.dump(found, f, indent=2)
print("Classes (saved to classes.json):")
for n, i in class_id.items():
    print(f"  {i} = {n}")

# 2. collect regions per image (the same image can be in several tasks)
regions, problems = {}, []
sizes = {}
for t in tasks:
    try:
        name = find_image(t["data"]["image"])
        if name is None:
            problems.append((t.get("id"), "image file not found in " + IMG_DIR)); continue
        if name not in sizes:
            img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                problems.append((name, "image unreadable")); continue
            sizes[name] = img.shape
        h, w = sizes[name]
        for r in brush_results(t):
            try:
                oh = r.get("original_height", h)
                ow = r.get("original_width", w)
                rgba = decode_rle(r["value"]["rle"]).reshape(oh, ow, 4)
                alpha = rgba[:, :, 3] > 0
                if (oh, ow) != (h, w):
                    alpha = cv2.resize(alpha.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST) > 0
                regions.setdefault(name, []).append(
                    (class_id[r["value"]["brushlabels"][0]], alpha))
            except Exception as e:
                problems.append((name, f"bad annotation: {e}"))
    except Exception as e:
        problems.append((t.get("id"), f"task failed: {e}"))

# 3. paint masks. Big regions first, small ones on top, so e.g. Caries stays
#    visible on top of a Teeth region instead of being overwritten.
for name, regs in regions.items():
    h, w = sizes[name]
    mask = np.zeros((h, w), dtype=np.uint8)
    for cid, alpha in sorted(regs, key=lambda x: -int(x[1].sum())):
        mask[alpha] = cid
    cv2.imwrite(os.path.join(OUT_DIR, name), mask)
    print("saved", name, np.unique(mask).tolist())

print(f"\n{len(regions)} masks saved, {len(problems)} problems")
for p in problems:
    print("  PROBLEM:", p)
