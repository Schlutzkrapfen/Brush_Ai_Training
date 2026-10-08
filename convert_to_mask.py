import json, os, cv2, numpy as np
from label_studio_converter.brush import decode_rle

CLASSES = ["Caries", "Teeth"]          # <- your label names; background is 0 automatically
class_id = {name: i + 1 for i, name in enumerate(CLASSES)}

with open("export.json") as f:
    tasks = json.load(f)

os.makedirs("data/masks", exist_ok=True)

for t in tasks:
    name = os.path.basename(t["data"]["image"]).split("-", 1)[-1]   # check this matches your image filenames
    img = cv2.imread(os.path.join("data/images", name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("image not found:", name); continue
    h, w = img.shape

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in t.get("annotations", []):
        for r in ann["result"]:
            if r.get("type") != "brushlabels":
                continue
            label = r["value"]["brushlabels"][0]
            if label not in class_id:
                print("unknown label:", label); continue
            rgba = decode_rle(r["value"]["rle"]).reshape(h, w, 4)
            mask[rgba[:, :, 3] > 0] = class_id[label]

    cv2.imwrite(f"data/masks/{name}", mask)
    print("saved", name, np.unique(mask))
