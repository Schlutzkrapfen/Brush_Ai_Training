import json, os, cv2, numpy as np
from label_studio_converter.brush import decode_rle   # pip install label-studio-converter

with open("export.json") as f:
    tasks = json.load(f)

os.makedirs("data/masks", exist_ok=True)

for t in tasks:
    # image name: strip Label Studio's random upload prefix
    img_path = t["data"]["image"]
    name = os.path.basename(img_path).split("-", 1)[-1]   # check this matches your image filenames
    img = cv2.imread(os.path.join("data/images", name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("image not found:", name); continue
    h, w = img.shape

    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in t.get("annotations", []):
        for r in ann["result"]:
            if r.get("type") == "brushlabels":
                rgba = decode_rle(r["value"]["rle"]).reshape(h, w, 4)
                mask[rgba[:, :, 3] > 0] = 255

    cv2.imwrite(f"data/masks/{name}", mask)
    print("saved", name)
